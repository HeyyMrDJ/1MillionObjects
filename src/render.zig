//! Instanced rendering with grid-based view culling.
//!
//! Each visible entity is sent to the GPU as a 20-byte instance (position, velocity,
//! swim phase) instead of raylib's 64-byte matrix. The vertex shader derives the
//! heading from the velocity and animates the swim wiggle. Every species is one
//! merged mesh, so the whole ocean takes one draw call per species.

const std = @import("std");
const rl = @import("raylib");
const rlgl = rl.gl;
const sim = @import("sim.zig");

pub const Instance = extern struct {
    x: f32,
    z: f32,
    vx: f32,
    vz: f32,
    phase: f32,
};

const RL_FLOAT = 0x1406;
const inst_loc_pos_vel = 14;
const inst_loc_phase = 15;
/// Instance buffers are rotated so we never overwrite one the GPU may still be reading.
const ring_size = 3;

pub const fish_length: f32 = 1.2;
pub const shark_length: f32 = 6.0;
/// Above this camera height fish are drawn with the low-poly mesh.
pub const lod_height: f32 = 140.0;

const vs_source =
    \\#version 330
    \\layout(location = 0) in vec3 vertexPosition;
    \\layout(location = 1) in vec2 vertexTexCoord;
    \\layout(location = 2) in vec3 vertexNormal;
    \\layout(location = 3) in vec4 vertexColor;
    \\layout(location = 14) in vec4 instPosVel;
    \\layout(location = 15) in float instPhase;
    \\uniform mat4 mvp;
    \\uniform float bodyLength;
    \\uniform float wiggleAmp;
    \\out vec2 fragTexCoord;
    \\out vec4 fragColor;
    \\out vec3 fragNormal;
    \\void main() {
    \\    vec3 p = vertexPosition;
    \\    // Lateral swim wave that grows toward the tail (model faces +Z).
    \\    float back = clamp(0.5 - p.z / bodyLength, 0.0, 1.0);
    \\    p.x += sin(instPhase - 6.0 * p.z / bodyLength) * wiggleAmp * bodyLength * back * back;
    \\    vec2 v = instPosVel.zw;
    \\    float len = length(v);
    \\    vec2 d = len > 1e-4 ? v / len : vec2(0.0, 1.0);
    \\    vec3 w = vec3(p.x * d.y + p.z * d.x, p.y, -p.x * d.x + p.z * d.y);
    \\    vec3 n = vertexNormal;
    \\    fragNormal = vec3(n.x * d.y + n.z * d.x, n.y, -n.x * d.x + n.z * d.y);
    \\    w.xz += instPosVel.xy;
    \\    fragTexCoord = vertexTexCoord;
    \\    fragColor = vertexColor;
    \\    gl_Position = mvp * vec4(w, 1.0);
    \\}
;

const fs_source =
    \\#version 330
    \\in vec2 fragTexCoord;
    \\in vec4 fragColor;
    \\in vec3 fragNormal;
    \\uniform sampler2D texture0;
    \\out vec4 finalColor;
    \\void main() {
    \\    vec4 base = texture(texture0, fragTexCoord) * fragColor;
    \\    // Two-sided: thin fins are drawn without backface culling.
    \\    float diff = abs(dot(normalize(fragNormal), normalize(vec3(0.3, 1.0, 0.2))));
    \\    finalColor = vec4(base.rgb * (0.4 + 0.6 * diff), 1.0);
    \\}
;

const Species = struct {
    mesh: rl.Mesh,
    texture_id: u32,
    length: f32,
    wiggle: f32,
    vbos: [ring_size]u32,

    fn initVbos(sp: *Species, capacity: usize) void {
        for (&sp.vbos) |*v| v.* = rlgl.rlLoadVertexBuffer(null, @intCast(capacity * @sizeOf(Instance)), true);
    }
};

const Row = struct { src_begin: u32, src_end: u32, dst: u32 };

pub const Stats = struct {
    visible_fish: usize = 0,
    visible_sharks: usize = 0,
    rows: usize = 0,
    low_poly: bool = false,
    fish_triangles: usize = 0,
    gather_ms: f32 = 0,
    upload_ms: f32 = 0,
};

pub const Renderer = struct {
    shader: rl.Shader,
    loc_mvp: i32,
    loc_body_length: i32,
    loc_wiggle: i32,
    loc_texture: i32,

    fish_hi: Species,
    fish_lo: Species,
    shark: Species,

    fish_instances: []Instance,
    shark_instances: []Instance,
    rows: [sim.grid_dim]Row = undefined,
    row_count: usize = 0,
    frame: usize = 0,
    stats: Stats = .{},

    pub fn init(gpa: std.mem.Allocator) !Renderer {
        const shader = try rl.loadShaderFromMemory(vs_source, fs_source);
        const white = rlgl.rlGetTextureIdDefault();

        const fish_model = rl.cdef.LoadModel("fish.glb");
        defer rl.unloadModel(fish_model);
        const shark_model = rl.cdef.LoadModel("shark.glb");
        defer rl.unloadModel(shark_model);
        if (fish_model.meshCount == 0 or shark_model.meshCount == 0) return error.LoadModel;

        var r: Renderer = .{
            .shader = shader,
            .loc_mvp = rl.getShaderLocation(shader, "mvp"),
            .loc_body_length = rl.getShaderLocation(shader, "bodyLength"),
            .loc_wiggle = rl.getShaderLocation(shader, "wiggleAmp"),
            .loc_texture = rl.getShaderLocation(shader, "texture0"),
            .fish_hi = .{ .mesh = mergeModel(fish_model, fish_length, false), .texture_id = white, .length = fish_length, .wiggle = 0.12, .vbos = undefined },
            .fish_lo = .{ .mesh = lowPolyFish(fish_length), .texture_id = white, .length = fish_length, .wiggle = 0.12, .vbos = undefined },
            .shark = .{ .mesh = mergeModel(shark_model, shark_length, false), .texture_id = modelTexture(shark_model, white), .length = shark_length, .wiggle = 0.06, .vbos = undefined },
            .fish_instances = try gpa.alloc(Instance, sim.fish_count),
            .shark_instances = try gpa.alloc(Instance, sim.shark_count),
        };
        // Both fish LODs read from the same instance data, so they share buffers.
        r.fish_hi.initVbos(sim.fish_count);
        r.fish_lo.vbos = r.fish_hi.vbos;
        r.shark.initVbos(sim.shark_count);
        return r;
    }

    /// Culls and packs visible instances, then draws them. Must be called inside BeginMode3D.
    pub fn draw(r: *Renderer, s: *sim.Sim, camera: rl.Camera3D) void {
        const t0 = s.now();
        const view = groundFootprint(camera);
        r.buildRows(s, view);
        const fish_n: usize = if (r.row_count == 0) 0 else r.rows[r.row_count - 1].dst + (r.rows[r.row_count - 1].src_end - r.rows[r.row_count - 1].src_begin);
        s.parallelFor(fish_n, gatherTask, .{ r, s });
        const shark_n = r.gatherSharks(s, view);
        const t1 = s.now();

        const low_poly = camera.position.y > lod_height;
        const fish = if (low_poly) &r.fish_lo else &r.fish_hi;
        const slot = r.frame % ring_size;
        r.frame += 1;

        rlgl.rlDrawRenderBatchActive();
        const mvp = rl.math.matrixMultiply(rlgl.rlGetMatrixModelview(), rlgl.rlGetMatrixProjection());
        r.drawSpecies(fish, slot, r.fish_instances[0..fish_n], mvp);
        r.drawSpecies(&r.shark, slot, r.shark_instances[0..shark_n], mvp);
        const t2 = s.now();

        r.stats = .{
            .visible_fish = fish_n,
            .visible_sharks = shark_n,
            .rows = r.row_count,
            .low_poly = low_poly,
            .fish_triangles = fish_n * @as(usize, @intCast(fish.mesh.triangleCount)),
            .gather_ms = msBetween(t0, t1),
            .upload_ms = msBetween(t1, t2),
        };
    }

    fn drawSpecies(r: *Renderer, sp: *const Species, slot: usize, instances: []const Instance, mvp: rl.Matrix) void {
        if (instances.len == 0) return;
        const vbo = sp.vbos[slot];
        rlgl.rlUpdateVertexBuffer(vbo, instances.ptr, @intCast(instances.len * @sizeOf(Instance)), 0);

        rlgl.rlEnableShader(r.shader.id);
        rlgl.rlSetUniformMatrix(r.loc_mvp, mvp);
        rl.setShaderValue(r.shader, r.loc_body_length, &sp.length, .float);
        rl.setShaderValue(r.shader, r.loc_wiggle, &sp.wiggle, .float);
        const unit: i32 = 0;
        rl.setShaderValue(r.shader, r.loc_texture, &unit, .int);
        rlgl.rlEnableShader(r.shader.id);
        rlgl.rlActiveTextureSlot(0);
        rlgl.rlEnableTexture(sp.texture_id);

        _ = rlgl.rlEnableVertexArray(@intCast(sp.mesh.vaoId));
        rlgl.rlEnableVertexBuffer(vbo);
        rlgl.rlSetVertexAttribute(inst_loc_pos_vel, 4, RL_FLOAT, false, @sizeOf(Instance), 0);
        rlgl.rlEnableVertexAttribute(inst_loc_pos_vel);
        rlgl.rlSetVertexAttributeDivisor(inst_loc_pos_vel, 1);
        rlgl.rlSetVertexAttribute(inst_loc_phase, 1, RL_FLOAT, false, @sizeOf(Instance), 16);
        rlgl.rlEnableVertexAttribute(inst_loc_phase);
        rlgl.rlSetVertexAttributeDivisor(inst_loc_phase, 1);

        rlgl.rlDisableBackfaceCulling();
        rlgl.rlDrawVertexArrayElementsInstanced(0, sp.mesh.triangleCount * 3, null, @intCast(instances.len));
        rlgl.rlEnableBackfaceCulling();

        rlgl.rlDisableVertexArray();
        rlgl.rlDisableVertexBuffer();
        rlgl.rlDisableTexture();
        rlgl.rlDisableShader();
    }

    /// For each grid row the view touches, finds the visible column span. Fish are sorted
    /// by cell in row-major order, so each span is one contiguous range of the SoA arrays.
    fn buildRows(r: *Renderer, s: *const sim.Sim, view: Footprint) void {
        r.row_count = 0;
        const margin = shark_length; // generous: covers model extents and one frame of motion
        const z_lo = cellIndex(view.min_z - margin) orelse 0;
        const z_hi = cellIndex(view.max_z + margin) orelse sim.grid_dim - 1;
        if (view.max_z + margin < 0 or view.min_z - margin >= sim.world_size) return;
        var dst: u32 = 0;
        for (z_lo..z_hi + 1) |row_z| {
            const band_lo = @as(f32, @floatFromInt(row_z)) * sim.cell_size - margin;
            const band_hi = band_lo + sim.cell_size + 2 * margin;
            const span = view.xRangeInBand(band_lo, band_hi) orelse continue;
            if (span[1] < 0 or span[0] >= sim.world_size) continue;
            const c0 = cellIndex(span[0] - margin) orelse 0;
            const c1 = cellIndex(span[1] + margin) orelse sim.grid_dim - 1;
            const row = row_z * sim.grid_dim;
            const begin = s.cell_start[row + c0];
            const end = s.cell_start[row + c1 + 1];
            if (end == begin) continue;
            r.rows[r.row_count] = .{ .src_begin = begin, .src_end = end, .dst = dst };
            r.row_count += 1;
            dst += end - begin;
        }
    }

    fn gatherSharks(r: *Renderer, s: *const sim.Sim, view: Footprint) usize {
        const sh = s.sharks;
        const m = shark_length;
        var n: usize = 0;
        for (0..sim.shark_count) |i| {
            const x = sh.x[i];
            const z = sh.z[i];
            if (x < view.min_x - m or x > view.max_x + m or z < view.min_z - m or z > view.max_z + m) continue;
            r.shark_instances[n] = .{ .x = x, .z = z, .vx = sh.vx[i], .vz = sh.vz[i], .phase = sh.phase[i] };
            n += 1;
        }
        return n;
    }
};

/// Copies output slots `begin..end` from the SoA fish arrays into interleaved instances.
fn gatherTask(ctx: struct { *Renderer, *sim.Sim }, _: usize, begin: usize, end: usize) void {
    const r = ctx[0];
    const f = ctx[1].currentFish();
    const out = r.fish_instances;
    for (r.rows[0..r.row_count]) |row| {
        const row_end = row.dst + (row.src_end - row.src_begin);
        const lo = @max(begin, row.dst);
        const hi = @min(end, row_end);
        if (lo >= hi) continue;
        var src = row.src_begin + (lo - row.dst);
        for (lo..hi) |d| {
            out[d] = .{ .x = f.x[src], .z = f.z[src], .vx = f.vx[src], .vz = f.vz[src], .phase = f.phase[src] };
            src += 1;
        }
    }
}

fn cellIndex(p: f32) ?usize {
    if (p < 0) return null;
    if (p >= sim.world_size) return null;
    return @intFromFloat(p / sim.cell_size);
}

fn msBetween(a: std.Io.Timestamp, b: std.Io.Timestamp) f32 {
    return @as(f32, @floatFromInt(a.durationTo(b).toNanoseconds())) / 1_000_000.0;
}

/// The camera frustum's footprint on the y = 0 plane (a convex quad).
const Footprint = struct {
    pts: [4][2]f32,
    min_x: f32,
    max_x: f32,
    min_z: f32,
    max_z: f32,

    /// The x extent of the quad clipped to the band `z_lo <= z <= z_hi`.
    fn xRangeInBand(fp: Footprint, z_lo: f32, z_hi: f32) ?[2]f32 {
        var lo = std.math.inf(f32);
        var hi = -std.math.inf(f32);
        for (0..4) |i| {
            const a = fp.pts[i];
            const b = fp.pts[(i + 1) % 4];
            var t0: f32 = 0;
            var t1: f32 = 1;
            const dz = b[1] - a[1];
            if (@abs(dz) < 1e-6) {
                if (a[1] < z_lo or a[1] > z_hi) continue;
            } else {
                const ta = (z_lo - a[1]) / dz;
                const tb = (z_hi - a[1]) / dz;
                t0 = @max(t0, @min(ta, tb));
                t1 = @min(t1, @max(ta, tb));
                if (t0 > t1) continue;
            }
            for ([_]f32{ t0, t1 }) |t| {
                const x = a[0] + (b[0] - a[0]) * t;
                lo = @min(lo, x);
                hi = @max(hi, x);
            }
        }
        return if (lo <= hi) .{ lo, hi } else null;
    }
};

fn groundFootprint(camera: rl.Camera3D) Footprint {
    const w: f32 = @floatFromInt(rl.getScreenWidth());
    const h: f32 = @floatFromInt(rl.getScreenHeight());
    const corners = [4]rl.Vector2{ .{ .x = 0, .y = 0 }, .{ .x = w, .y = 0 }, .{ .x = w, .y = h }, .{ .x = 0, .y = h } };
    var fp: Footprint = .{ .pts = undefined, .min_x = std.math.inf(f32), .max_x = -std.math.inf(f32), .min_z = std.math.inf(f32), .max_z = -std.math.inf(f32) };
    for (corners, 0..) |c, i| {
        const ray = rl.getScreenToWorldRay(c, camera);
        // Rays at or above the horizon never hit the ground; push them out far instead.
        const t = if (ray.direction.y < -1e-3) -ray.position.y / ray.direction.y else 1e5;
        const p = [2]f32{ ray.position.x + ray.direction.x * t, ray.position.z + ray.direction.z * t };
        fp.pts[i] = p;
        fp.min_x = @min(fp.min_x, p[0]);
        fp.max_x = @max(fp.max_x, p[0]);
        fp.min_z = @min(fp.min_z, p[1]);
        fp.max_z = @max(fp.max_z, p[1]);
    }
    return fp;
}

fn modelTexture(model: rl.Model, fallback: u32) u32 {
    const mat = model.materials[@intCast(model.meshMaterial[0])];
    const id = mat.maps[0].texture.id;
    return if (id != 0) @intCast(id) else fallback;
}

/// Merges all meshes of a model into one indexed mesh centred on the origin, scaled to
/// `length` along Z, with each mesh's material colour baked into vertex colours.
fn mergeModel(model: rl.Model, length: f32, flip: bool) rl.Mesh {
    const meshes = model.meshes[0..@intCast(model.meshCount)];
    var bb_min = [3]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var bb_max = [3]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    var vert_total: usize = 0;
    var index_total: usize = 0;
    for (meshes) |m| {
        const n: usize = @intCast(m.vertexCount);
        for (0..n) |v| for (0..3) |k| {
            bb_min[k] = @min(bb_min[k], m.vertices[v * 3 + k]);
            bb_max[k] = @max(bb_max[k], m.vertices[v * 3 + k]);
        };
        vert_total += n;
        index_total += if (m.indices != null) @as(usize, @intCast(m.triangleCount)) * 3 else n;
    }
    std.debug.assert(vert_total <= std.math.maxInt(u16));
    const scale = length / (bb_max[2] - bb_min[2]);
    const center = [3]f32{ (bb_min[0] + bb_max[0]) / 2, (bb_min[1] + bb_max[1]) / 2, (bb_min[2] + bb_max[2]) / 2 };
    const sign: f32 = if (flip) -1 else 1;

    var out = std.mem.zeroes(rl.Mesh);
    out.vertexCount = @intCast(vert_total);
    out.triangleCount = @intCast(index_total / 3);
    out.vertices = cAlloc(f32, vert_total * 3);
    out.normals = cAlloc(f32, vert_total * 3);
    out.texcoords = cAlloc(f32, vert_total * 2);
    out.colors = cAlloc(u8, vert_total * 4);
    out.indices = cAlloc(c_ushort, index_total);

    var vbase: usize = 0;
    var ibase: usize = 0;
    for (meshes, 0..) |m, mi| {
        const n: usize = @intCast(m.vertexCount);
        const color = model.materials[@intCast(model.meshMaterial[mi])].maps[0].color;
        for (0..n) |v| {
            const o = vbase + v;
            // Flipping rotates 180 degrees about Y (negate X and Z), which keeps winding intact.
            out.vertices[o * 3 + 0] = (m.vertices[v * 3 + 0] - center[0]) * scale * sign;
            out.vertices[o * 3 + 1] = (m.vertices[v * 3 + 1] - center[1]) * scale;
            out.vertices[o * 3 + 2] = (m.vertices[v * 3 + 2] - center[2]) * scale * sign;
            if (m.normals != null) {
                out.normals[o * 3 + 0] = m.normals[v * 3 + 0] * sign;
                out.normals[o * 3 + 1] = m.normals[v * 3 + 1];
                out.normals[o * 3 + 2] = m.normals[v * 3 + 2] * sign;
            } else {
                out.normals[o * 3 + 1] = 1;
            }
            out.texcoords[o * 2 + 0] = if (m.texcoords != null) m.texcoords[v * 2 + 0] else 0;
            out.texcoords[o * 2 + 1] = if (m.texcoords != null) m.texcoords[v * 2 + 1] else 0;
            out.colors[o * 4 + 0] = color.r;
            out.colors[o * 4 + 1] = color.g;
            out.colors[o * 4 + 2] = color.b;
            out.colors[o * 4 + 3] = 255;
        }
        if (m.indices != null) {
            const ni: usize = @as(usize, @intCast(m.triangleCount)) * 3;
            for (0..ni) |k| out.indices[ibase + k] = @intCast(m.indices[k] + vbase);
            ibase += ni;
        } else {
            for (0..n) |k| out.indices[ibase + k] = @intCast(vbase + k);
            ibase += n;
        }
        vbase += n;
    }
    rl.uploadMesh(&out, false);
    return out;
}

/// A ~12-triangle fish for far-away views, where the full model would be sub-pixel anyway.
fn lowPolyFish(length: f32) rl.Mesh {
    const L = length;
    const nose = [3]f32{ 0, 0, 0.5 * L };
    const left = [3]f32{ -0.13 * L, 0, 0.05 * L };
    const right = [3]f32{ 0.13 * L, 0, 0.05 * L };
    const top = [3]f32{ 0, 0.12 * L, 0.05 * L };
    const bottom = [3]f32{ 0, -0.1 * L, 0.05 * L };
    const tail = [3]f32{ 0, 0, -0.3 * L };
    const fin_l = [3]f32{ -0.15 * L, 0, -0.5 * L };
    const fin_r = [3]f32{ 0.15 * L, 0, -0.5 * L };
    const body = [3]u8{ 220, 70, 20 };
    const fin = [3]u8{ 240, 200, 150 };
    const tris = [_]struct { [3][3]f32, [3]u8 }{
        .{ .{ nose, right, top }, body },   .{ .{ nose, top, left }, body },
        .{ .{ nose, bottom, right }, body }, .{ .{ nose, left, bottom }, body },
        .{ .{ tail, top, right }, body },   .{ .{ tail, left, top }, body },
        .{ .{ tail, right, bottom }, body }, .{ .{ tail, bottom, left }, body },
        // Tail fin, both windings so it shows from above and below.
        .{ .{ tail, fin_r, fin_l }, fin },  .{ .{ tail, fin_l, fin_r }, fin },
    };
    const n = tris.len * 3;
    var out = std.mem.zeroes(rl.Mesh);
    out.vertexCount = n;
    out.triangleCount = tris.len;
    out.vertices = cAlloc(f32, n * 3);
    out.normals = cAlloc(f32, n * 3);
    out.texcoords = cAlloc(f32, n * 2);
    out.colors = cAlloc(u8, n * 4);
    out.indices = cAlloc(c_ushort, n);
    for (tris, 0..) |tri, t| {
        const p = tri[0];
        const e1 = [3]f32{ p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2] };
        const e2 = [3]f32{ p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2] };
        var nrm = [3]f32{ e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
        const len = @sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
        for (&nrm) |*c| c.* /= len;
        for (0..3) |k| {
            const v = t * 3 + k;
            for (0..3) |a| {
                out.vertices[v * 3 + a] = p[k][a];
                out.normals[v * 3 + a] = nrm[a];
                out.colors[v * 4 + a] = tri[1][a];
            }
            out.colors[v * 4 + 3] = 255;
            out.indices[v] = @intCast(v);
        }
    }
    rl.uploadMesh(&out, false);
    return out;
}

/// raylib frees mesh arrays with the C allocator, so allocate them with it too.
fn cAlloc(comptime T: type, n: usize) [*c]T {
    const s = std.heap.c_allocator.alloc(T, n) catch @panic("out of memory");
    @memset(s, std.mem.zeroes(T));
    return s.ptr;
}
