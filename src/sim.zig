//! Simulation state and update. Everything is stored as structs of arrays and
//! the fish are re-sorted into spatial grid order every frame with a parallel
//! counting sort, so neighbour queries and view culling read contiguous memory.
//!
//! The sort keys on each fish's cell at the start of the frame, which lets the
//! update and the scatter happen in one pass: each fish is read once, updated,
//! and written straight to its sorted slot. Fish move well under one unit per
//! frame, so the grid is never more than a fraction of a unit stale.

const std = @import("std");
const Io = std.Io;

pub const shark_count: usize = 1000;
pub const fish_count: usize = 10_000_000;
/// Sized for roughly 0.25 fish per square unit.
pub const world_size: f32 = 6400.0;

/// Fish grid: used for shark prey searches, fish threat lookups and view culling.
/// 32 keeps the cell count (200 x 200) within the u16 used for `fish_cell`.
pub const cell_size: f32 = 32.0;
pub const grid_dim: usize = @intFromFloat(world_size / cell_size);
pub const cell_count: usize = grid_dim * grid_dim;
const flow_dim: usize = grid_dim + 1;
comptime {
    std.debug.assert(cell_count <= std.math.maxInt(u16)); // fish_cell is u16
}

const fish_cruise_speed: f32 = 2.5;
const fish_flee_speed: f32 = 5.5;
const fish_accel: f32 = 3.0;
const flee_radius: f32 = 20.0;
const shark_chase_speed: f32 = 7.0;
const shark_cruise_speed: f32 = 3.5;
const shark_accel: f32 = 1.5;
const eat_radius: f32 = 1.0;
const wall_margin: f32 = 30.0;
const max_edge: f32 = world_size - 0.01;

const no_prey = std.math.maxInt(u32);
/// A shark's flee circle spans at most this many cells per axis.
const threat_cells_per_axis: usize = @as(usize, @intFromFloat(2 * flee_radius / cell_size)) + 2;

/// One set of per-entity arrays (structure of arrays). Positions need full precision
/// across the world; velocity and phase fit in half floats, which keeps a fish at
/// 14 bytes and cuts memory traffic in both the update and the GPU upload.
pub const Bodies = struct {
    x: []f32,
    z: []f32,
    vx: []f16,
    vz: []f16,
    /// Swim animation phase in radians, advanced by speed.
    phase: []f16,

    pub fn alloc(gpa: std.mem.Allocator, n: usize) !Bodies {
        return .{
            .x = try gpa.alloc(f32, n),
            .z = try gpa.alloc(f32, n),
            .vx = try gpa.alloc(f16, n),
            .vz = try gpa.alloc(f16, n),
            .phase = try gpa.alloc(f16, n),
        };
    }

    fn free(b: Bodies, gpa: std.mem.Allocator) void {
        gpa.free(b.x);
        gpa.free(b.z);
        gpa.free(b.vx);
        gpa.free(b.vz);
        gpa.free(b.phase);
    }
};

/// Milliseconds spent in each phase of the last step.
pub const Timings = struct {
    flow: f32 = 0,
    fish: f32 = 0,
    sort: f32 = 0,
    sharks: f32 = 0,
    total: f32 = 0,
};

pub const Sim = struct {
    gpa: std.mem.Allocator,
    io: Io,
    task_count: usize,

    /// Double-buffered: `fish[cur]` is sorted by cell, the other is the scatter target.
    fish: [2]Bodies,
    cur: usize = 0,
    fish_cell: []u16,
    /// Per-task cell histograms, later turned into per-task scatter cursors.
    hist: []u32,
    /// `cell_start[c]..cell_start[c+1]` is the range of fish in cell `c` (row-major).
    cell_start: []u32,

    /// Current direction at each cell corner (`flow_dim` x `flow_dim` nodes).
    flow_x: []f32,
    flow_z: []f32,

    sharks: Bodies,
    /// `threat_start[c]..threat_start[c+1]` holds the sharks whose flee circle overlaps cell `c`.
    threat_start: []u32,
    threat_cursor: []u32,
    threat_x: []f32,
    threat_z: []f32,
    shark_prey: []u32,

    time: f32 = 0,
    dt: f32 = 0,
    eaten_total: u64 = 0,
    rng: std.Random.DefaultPrng,
    timings: Timings = .{},

    pub fn create(gpa: std.mem.Allocator, io: Io, task_count: usize) !*Sim {
        const max_threats = shark_count * threat_cells_per_axis * threat_cells_per_axis;
        const s = try gpa.create(Sim);
        s.* = .{
            .gpa = gpa,
            .io = io,
            .task_count = task_count,
            .fish = .{ try Bodies.alloc(gpa, fish_count), try Bodies.alloc(gpa, fish_count) },
            .fish_cell = try gpa.alloc(u16, fish_count),
            .hist = try gpa.alloc(u32, task_count * cell_count),
            .cell_start = try gpa.alloc(u32, cell_count + 1),
            .flow_x = try gpa.alloc(f32, flow_dim * flow_dim),
            .flow_z = try gpa.alloc(f32, flow_dim * flow_dim),
            .sharks = try Bodies.alloc(gpa, shark_count),
            .threat_start = try gpa.alloc(u32, cell_count + 1),
            .threat_cursor = try gpa.alloc(u32, cell_count),
            .threat_x = try gpa.alloc(f32, max_threats),
            .threat_z = try gpa.alloc(f32, max_threats),
            .shark_prey = try gpa.alloc(u32, shark_count),
            .rng = .init(0x1_000_000),
        };

        const r = s.rng.random();
        const f = s.fish[0];
        for (0..fish_count) |i| {
            f.x[i] = r.float(f32) * max_edge;
            f.z[i] = r.float(f32) * max_edge;
            const a = r.float(f32) * std.math.tau;
            f.vx[i] = @floatCast(@cos(a) * fish_cruise_speed);
            f.vz[i] = @floatCast(@sin(a) * fish_cruise_speed);
            f.phase[i] = @floatCast(r.float(f32) * std.math.tau);
        }
        for (0..shark_count) |i| {
            s.sharks.x[i] = r.float(f32) * max_edge;
            s.sharks.z[i] = r.float(f32) * max_edge;
            s.sharks.vx[i] = 0;
            s.sharks.vz[i] = 0;
            s.sharks.phase[i] = @floatCast(r.float(f32) * std.math.tau);
        }

        // A zero-length step sorts the fish so the grid is valid before the first frame.
        s.step(0);
        return s;
    }

    pub fn destroy(s: *Sim) void {
        const gpa = s.gpa;
        s.fish[0].free(gpa);
        s.fish[1].free(gpa);
        s.sharks.free(gpa);
        gpa.free(s.fish_cell);
        gpa.free(s.hist);
        gpa.free(s.cell_start);
        gpa.free(s.flow_x);
        gpa.free(s.flow_z);
        gpa.free(s.threat_start);
        gpa.free(s.threat_cursor);
        gpa.free(s.threat_x);
        gpa.free(s.threat_z);
        gpa.free(s.shark_prey);
        gpa.destroy(s);
    }

    pub fn currentFish(s: *const Sim) Bodies {
        return s.fish[s.cur];
    }

    pub fn step(s: *Sim, dt: f32) void {
        s.dt = dt;
        s.time += dt;
        const t0 = s.now();

        s.parallelFor(flow_dim, flowTask, s);
        s.buildThreats();
        const t1 = s.now();

        s.parallelFor(fish_count, cellTask, s);
        s.parallelFor(cell_count, cellTotalsTask, s);
        s.cell_start[0] = 0;
        for (1..cell_count + 1) |c| s.cell_start[c] += s.cell_start[c - 1];
        s.parallelFor(cell_count, cellCursorsTask, s);
        const t2 = s.now();

        s.parallelFor(fish_count, fishTask, s);
        s.cur ^= 1;
        const t3 = s.now();

        s.parallelFor(shark_count, sharkTask, s);
        s.applyEats();
        const t4 = s.now();

        s.timings = .{
            .flow = ms(t0, t1),
            .sort = ms(t1, t2),
            .fish = ms(t2, t3),
            .sharks = ms(t3, t4),
            .total = ms(t0, t4),
        };
    }

    /// Splits `0..n` into one contiguous chunk per task and runs them on the Io thread pool.
    /// `func` is called as `func(ctx, task_index, begin, end)`. Chunk boundaries depend only
    /// on `n` and the task count, so two calls with the same `n` get identical chunks.
    pub fn parallelFor(s: *Sim, n: usize, comptime func: anytype, ctx: anytype) void {
        var group: Io.Group = .init;
        const tasks = s.task_count;
        for (0..tasks) |t| {
            group.async(s.io, func, .{ ctx, t, n * t / tasks, n * (t + 1) / tasks });
        }
        group.await(s.io) catch {};
    }

    pub fn now(s: *const Sim) Io.Timestamp {
        return Io.Timestamp.now(s.io, .awake);
    }

    /// Lists each shark in every cell its flee circle overlaps, so a fish only has to scan
    /// its own cell's list. For most fish that list is empty.
    fn buildThreats(s: *Sim) void {
        const sh = s.sharks;
        const start = s.threat_start;
        @memset(start, 0);
        for (0..shark_count) |i| {
            const r = threatCells(sh.x[i], sh.z[i]);
            for (r.z0..r.z1 + 1) |cz| for (r.x0..r.x1 + 1) |cx| {
                start[cz * grid_dim + cx + 1] += 1;
            };
        }
        for (1..cell_count + 1) |c| start[c] += start[c - 1];
        @memcpy(s.threat_cursor, start[0..cell_count]);
        for (0..shark_count) |i| {
            const r = threatCells(sh.x[i], sh.z[i]);
            for (r.z0..r.z1 + 1) |cz| for (r.x0..r.x1 + 1) |cx| {
                const c = cz * grid_dim + cx;
                const d = s.threat_cursor[c];
                s.threat_cursor[c] = d + 1;
                s.threat_x[d] = sh.x[i];
                s.threat_z[d] = sh.z[i];
            };
        }
    }

    fn applyEats(s: *Sim) void {
        const f = s.fish[s.cur];
        const r = s.rng.random();
        for (s.shark_prey) |prey| {
            if (prey == no_prey) continue;
            // Respawn somewhere else so the population stays constant. The fish stays in
            // its old grid slot until next frame's sort, which is harmless.
            f.x[prey] = r.float(f32) * max_edge;
            f.z[prey] = r.float(f32) * max_edge;
            s.eaten_total += 1;
        }
    }
};

fn ms(a: Io.Timestamp, b: Io.Timestamp) f32 {
    return @as(f32, @floatFromInt(a.durationTo(b).toNanoseconds())) / 1_000_000.0;
}

pub fn fishCell(x: f32, z: f32) u32 {
    const cx: u32 = @intFromFloat(x * (1.0 / cell_size));
    const cz: u32 = @intFromFloat(z * (1.0 / cell_size));
    return cz * @as(u32, grid_dim) + cx;
}

const CellRange = struct { x0: usize, x1: usize, z0: usize, z1: usize };

/// The cells overlapped by the square around a shark's flee circle.
fn threatCells(x: f32, z: f32) CellRange {
    return .{
        .x0 = @intFromFloat(@max(x - flee_radius, 0) * (1.0 / cell_size)),
        .x1 = @intFromFloat(@min(x + flee_radius, max_edge) * (1.0 / cell_size)),
        .z0 = @intFromFloat(@max(z - flee_radius, 0) * (1.0 / cell_size)),
        .z1 = @intFromFloat(@min(z + flee_radius, max_edge) * (1.0 / cell_size)),
    };
}

/// Ocean currents: the curl of a sum of moving sine waves, sampled at cell corners. Being
/// divergence-free, it moves fish around in swirls without piling them up in sinks.
fn flowTask(s: *Sim, _: usize, begin: usize, end: usize) void {
    const t = s.time;
    for (begin..end) |nz| {
        const z = @as(f32, @floatFromInt(nz)) * cell_size;
        for (0..flow_dim) |nx| {
            const x = @as(f32, @floatFromInt(nx)) * cell_size;
            const k = 0.015;
            const a1 = k * x + 0.07 * t;
            const b1 = 1.3 * k * z;
            const m = 0.009;
            const a2 = m * (0.8 * x + 0.6 * z) - 0.05 * t;
            const n = 0.031;
            const a3 = n * z + 0.11 * t;
            const b3 = 0.7 * n * x - 0.06 * t;
            const dpsi_dx = @cos(a1) * @cos(b1) + 0.64 * @cos(a2) + 0.28 * @sin(a3) * @cos(b3);
            const dpsi_dz = -1.3 * @sin(a1) * @sin(b1) + 0.48 * @cos(a2) + 0.4 * @cos(a3) * @sin(b3);
            s.flow_x[nz * flow_dim + nx] = dpsi_dz;
            s.flow_z[nz * flow_dim + nx] = -dpsi_dx;
        }
    }
}

/// Bilinear sample of the current at a position inside cell `c`. Interpolating matters: a
/// per-cell constant current shears at cell borders and draws visible grid lines of fish.
fn sampleFlow(s: *const Sim, x: f32, z: f32, c: usize) [2]f32 {
    const iz = c / grid_dim;
    const ix = c - iz * grid_dim;
    const fx = std.math.clamp(x * (1.0 / cell_size) - @as(f32, @floatFromInt(ix)), 0, 1);
    const fz = std.math.clamp(z * (1.0 / cell_size) - @as(f32, @floatFromInt(iz)), 0, 1);
    const n00 = iz * flow_dim + ix;
    const n01 = n00 + flow_dim;
    const top_x = s.flow_x[n00] + (s.flow_x[n00 + 1] - s.flow_x[n00]) * fx;
    const top_z = s.flow_z[n00] + (s.flow_z[n00 + 1] - s.flow_z[n00]) * fx;
    const bot_x = s.flow_x[n01] + (s.flow_x[n01 + 1] - s.flow_x[n01]) * fx;
    const bot_z = s.flow_z[n01] + (s.flow_z[n01 + 1] - s.flow_z[n01]) * fx;
    return .{ top_x + (bot_x - top_x) * fz, top_z + (bot_z - top_z) * fz };
}

fn wallPush(p: f32) f32 {
    if (p < wall_margin) return (wall_margin - p) / wall_margin;
    if (p > world_size - wall_margin) return -(p - (world_size - wall_margin)) / wall_margin;
    return 0;
}

/// Pass 1 of the sort: bucket every fish by its current cell.
fn cellTask(s: *Sim, task: usize, begin: usize, end: usize) void {
    const f = s.fish[s.cur];
    const hist = s.hist[task * cell_count ..][0..cell_count];
    @memset(hist, 0);
    for (begin..end) |i| {
        const c = fishCell(f.x[i], f.z[i]);
        s.fish_cell[i] = @intCast(c);
        hist[c] += 1;
    }
}

/// Updates each fish and writes it to its sorted slot. Uses the same chunks as
/// `cellTask`, so each task consumes exactly the cursors its histogram produced.
fn fishTask(s: *Sim, task: usize, begin: usize, end: usize) void {
    const f = s.fish[s.cur];
    const dst = s.fish[s.cur ^ 1];
    const cursor = s.hist[task * cell_count ..][0..cell_count];
    const dt = s.dt;
    const steer = @min(1.0, fish_accel * dt);
    const flee_r2 = flee_radius * flee_radius;

    for (begin..end) |i| {
        const c = s.fish_cell[i];
        var x = f.x[i];
        var z = f.z[i];
        var vx: f32 = f.vx[i];
        var vz: f32 = f.vz[i];

        // Cruise along the current at a roughly constant speed.
        const flow = sampleFlow(s, x, z, c);
        const flow_x = flow[0];
        const flow_z = flow[1];
        const flow_scale = fish_cruise_speed / @max(@sqrt(flow_x * flow_x + flow_z * flow_z), 0.5);
        var dx = flow_x * flow_scale + wallPush(x) * fish_cruise_speed * 2;
        var dz = flow_z * flow_scale + wallPush(z) * fish_cruise_speed * 2;

        // Flee the nearest shark that can reach this cell.
        var best_d2: f32 = flee_r2;
        var away_x: f32 = 0;
        var away_z: f32 = 0;
        for (s.threat_start[c]..s.threat_start[c + 1]) |k| {
            const ax = x - s.threat_x[k];
            const az = z - s.threat_z[k];
            const d2 = ax * ax + az * az;
            if (d2 < best_d2) {
                best_d2 = d2;
                away_x = ax;
                away_z = az;
            }
        }
        if (best_d2 < flee_r2) {
            const d = @sqrt(best_d2) + 1e-4;
            const urgency = 1.0 - d / flee_radius;
            const panic = fish_flee_speed / d;
            dx += (away_x * panic - dx) * (0.5 + 0.5 * urgency);
            dz += (away_z * panic - dz) * (0.5 + 0.5 * urgency);
        }

        vx += (dx - vx) * steer;
        vz += (dz - vz) * steer;
        const speed2 = vx * vx + vz * vz;
        if (speed2 > fish_flee_speed * fish_flee_speed) {
            const k = fish_flee_speed / @sqrt(speed2);
            vx *= k;
            vz *= k;
        }
        x = std.math.clamp(x + vx * dt, 0, max_edge);
        z = std.math.clamp(z + vz * dt, 0, max_edge);

        var phase = @as(f32, f.phase[i]) + dt * (3.0 + 2.0 * @sqrt(speed2));
        if (phase > std.math.tau) phase -= std.math.tau;

        const d = cursor[c];
        cursor[c] = d + 1;
        dst.x[d] = x;
        dst.z[d] = z;
        dst.vx[d] = @floatCast(vx);
        dst.vz[d] = @floatCast(vz);
        dst.phase[d] = @floatCast(phase);
    }
}

/// cell_start[c + 1] = number of fish in cell c (prefix-summed afterwards).
fn cellTotalsTask(s: *Sim, _: usize, begin: usize, end: usize) void {
    for (begin..end) |c| {
        var total: u32 = 0;
        for (0..s.task_count) |t| total += s.hist[t * cell_count + c];
        s.cell_start[c + 1] = total;
    }
}

/// Turns each task's histogram into the first destination slot that task writes in each cell.
fn cellCursorsTask(s: *Sim, _: usize, begin: usize, end: usize) void {
    for (begin..end) |c| {
        var run = s.cell_start[c];
        for (0..s.task_count) |t| {
            const h = &s.hist[t * cell_count + c];
            const n = h.*;
            h.* = run;
            run += n;
        }
    }
}

fn sharkTask(s: *Sim, _: usize, begin: usize, end: usize) void {
    const f = s.fish[s.cur];
    const sh = s.sharks;
    const dt = s.dt;
    const steer = @min(1.0, shark_accel * dt);

    for (begin..end) |i| {
        var x = sh.x[i];
        var z = sh.z[i];
        var vx: f32 = sh.vx[i];
        var vz: f32 = sh.vz[i];

        // Nearest fish in the 3x3 fish cells around us.
        const cx: usize = @intFromFloat(x * (1.0 / cell_size));
        const cz: usize = @intFromFloat(z * (1.0 / cell_size));
        const x0 = cx -| 1;
        const x1 = @min(cx + 1, grid_dim - 1);
        var best_d2 = std.math.inf(f32);
        var best: u32 = no_prey;
        for ((cz -| 1)..@min(cz + 1, grid_dim - 1) + 1) |row_z| {
            const row = row_z * grid_dim;
            // Cells in a row are contiguous, so scan the whole row segment at once.
            for (s.cell_start[row + x0]..s.cell_start[row + x1 + 1]) |k| {
                const ax = f.x[k] - x;
                const az = f.z[k] - z;
                const d2 = ax * ax + az * az;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best = @intCast(k);
                }
            }
        }

        var dx: f32 = undefined;
        var dz: f32 = undefined;
        if (best != no_prey) {
            const d = @sqrt(best_d2) + 1e-4;
            dx = (f.x[best] - x) / d * shark_chase_speed;
            dz = (f.z[best] - z) / d * shark_chase_speed;
        } else {
            const c = fishCell(x, z);
            const flow = sampleFlow(s, x, z, c);
            dx = flow[0] * shark_cruise_speed;
            dz = flow[1] * shark_cruise_speed;
        }
        dx += wallPush(x) * shark_chase_speed;
        dz += wallPush(z) * shark_chase_speed;
        s.shark_prey[i] = if (best_d2 < eat_radius * eat_radius) best else no_prey;

        vx += (dx - vx) * steer;
        vz += (dz - vz) * steer;
        const speed2 = vx * vx + vz * vz;
        if (speed2 > shark_chase_speed * shark_chase_speed) {
            const k = shark_chase_speed / @sqrt(speed2);
            vx *= k;
            vz *= k;
        }
        x = std.math.clamp(x + vx * dt, 0, max_edge);
        z = std.math.clamp(z + vz * dt, 0, max_edge);

        var phase = @as(f32, sh.phase[i]) + dt * (2.0 + 0.6 * @sqrt(speed2));
        if (phase > std.math.tau) phase -= std.math.tau;

        sh.x[i] = x;
        sh.z[i] = z;
        sh.vx[i] = @floatCast(vx);
        sh.vz[i] = @floatCast(vz);
        sh.phase[i] = @floatCast(phase);
    }
}
