//! Simulation state and update. Everything is stored as structs of arrays and
//! the fish are re-sorted into spatial grid order every frame with a parallel
//! counting sort, so neighbour queries and view culling read contiguous memory.

const std = @import("std");
const Io = std.Io;

pub const shark_count: usize = 1000;
pub const fish_count: usize = 1_000_000 - shark_count;
pub const world_size: f32 = 2000.0;

/// Fish grid: used by sharks to find prey and by the renderer for culling.
pub const cell_size: f32 = 16.0;
pub const grid_dim: usize = @intFromFloat(world_size / cell_size);
pub const cell_count: usize = grid_dim * grid_dim;

/// Shark grid: used by fish to find nearby sharks. Must be >= flee_radius.
const shark_cell_size: f32 = 25.0;
const shark_grid_dim: usize = @intFromFloat(world_size / shark_cell_size);
const shark_cell_count: usize = shark_grid_dim * shark_grid_dim;

/// The current field is sampled at grid nodes and bilinearly interpolated.
const flow_dim: usize = grid_dim + 1;

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

/// One set of per-entity arrays (structure of arrays).
pub const Bodies = struct {
    x: []f32,
    z: []f32,
    vx: []f32,
    vz: []f32,
    /// Swim animation phase in radians, advanced by speed.
    phase: []f32,

    fn alloc(gpa: std.mem.Allocator, n: usize) !Bodies {
        return .{
            .x = try gpa.alloc(f32, n),
            .z = try gpa.alloc(f32, n),
            .vx = try gpa.alloc(f32, n),
            .vz = try gpa.alloc(f32, n),
            .phase = try gpa.alloc(f32, n),
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
    fish_cell: []u32,
    /// Per-task cell histograms, later turned into per-task scatter cursors.
    hist: []u32,
    /// `cell_start[c]..cell_start[c+1]` is the range of fish in cell `c` (row-major).
    cell_start: []u32,

    flow_x: []f32,
    flow_z: []f32,

    sharks: Bodies,
    shark_cell_start: []u32,
    shark_cursor: []u32,
    shark_sorted_x: []f32,
    shark_sorted_z: []f32,
    shark_prey: []u32,

    time: f32 = 0,
    dt: f32 = 0,
    eaten_total: u64 = 0,
    rng: std.Random.DefaultPrng,
    timings: Timings = .{},

    pub fn create(gpa: std.mem.Allocator, io: Io, task_count: usize) !*Sim {
        const s = try gpa.create(Sim);
        s.* = .{
            .gpa = gpa,
            .io = io,
            .task_count = task_count,
            .fish = .{ try Bodies.alloc(gpa, fish_count), try Bodies.alloc(gpa, fish_count) },
            .fish_cell = try gpa.alloc(u32, fish_count),
            .hist = try gpa.alloc(u32, task_count * cell_count),
            .cell_start = try gpa.alloc(u32, cell_count + 1),
            .flow_x = try gpa.alloc(f32, flow_dim * flow_dim),
            .flow_z = try gpa.alloc(f32, flow_dim * flow_dim),
            .sharks = try Bodies.alloc(gpa, shark_count),
            .shark_cell_start = try gpa.alloc(u32, shark_cell_count + 1),
            .shark_cursor = try gpa.alloc(u32, shark_cell_count),
            .shark_sorted_x = try gpa.alloc(f32, shark_count),
            .shark_sorted_z = try gpa.alloc(f32, shark_count),
            .shark_prey = try gpa.alloc(u32, shark_count),
            .rng = .init(0x1_000_000),
        };

        const r = s.rng.random();
        const f = s.fish[0];
        for (0..fish_count) |i| {
            f.x[i] = r.float(f32) * max_edge;
            f.z[i] = r.float(f32) * max_edge;
            const a = r.float(f32) * std.math.tau;
            f.vx[i] = @cos(a) * fish_cruise_speed;
            f.vz[i] = @sin(a) * fish_cruise_speed;
            f.phase[i] = r.float(f32) * std.math.tau;
        }
        for (0..shark_count) |i| {
            s.sharks.x[i] = r.float(f32) * max_edge;
            s.sharks.z[i] = r.float(f32) * max_edge;
            s.sharks.vx[i] = 0;
            s.sharks.vz[i] = 0;
            s.sharks.phase[i] = r.float(f32) * std.math.tau;
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
        gpa.free(s.shark_cell_start);
        gpa.free(s.shark_cursor);
        gpa.free(s.shark_sorted_x);
        gpa.free(s.shark_sorted_z);
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
        const t1 = s.now();

        s.buildSharkGrid();
        s.parallelFor(fish_count, fishTask, s);
        const t2 = s.now();

        s.parallelFor(cell_count, cellTotalsTask, s);
        s.cell_start[0] = 0;
        for (1..cell_count + 1) |c| s.cell_start[c] += s.cell_start[c - 1];
        s.parallelFor(cell_count, cellCursorsTask, s);
        s.parallelFor(fish_count, scatterTask, s);
        s.cur ^= 1;
        const t3 = s.now();

        s.parallelFor(shark_count, sharkTask, s);
        s.applyEats();
        const t4 = s.now();

        s.timings = .{
            .flow = ms(t0, t1),
            .fish = ms(t1, t2),
            .sort = ms(t2, t3),
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

    fn buildSharkGrid(s: *Sim) void {
        const sh = s.sharks;
        const start = s.shark_cell_start;
        @memset(start, 0);
        for (0..shark_count) |i| start[sharkCell(sh.x[i], sh.z[i]) + 1] += 1;
        for (1..shark_cell_count + 1) |c| start[c] += start[c - 1];
        @memcpy(s.shark_cursor, start[0..shark_cell_count]);
        for (0..shark_count) |i| {
            const c = sharkCell(sh.x[i], sh.z[i]);
            const d = s.shark_cursor[c];
            s.shark_cursor[c] = d + 1;
            s.shark_sorted_x[d] = sh.x[i];
            s.shark_sorted_z[d] = sh.z[i];
        }
    }

    fn applyEats(s: *Sim) void {
        const f = s.fish[s.cur];
        const r = s.rng.random();
        for (s.shark_prey) |prey| {
            if (prey == no_prey) continue;
            // Respawn somewhere else so the population stays at one million. The fish
            // stays in its old grid slot until next frame's sort, which is harmless.
            f.x[prey] = r.float(f32) * max_edge;
            f.z[prey] = r.float(f32) * max_edge;
            s.eaten_total += 1;
        }
    }

    /// Bilinear sample of the current field at a world position.
    fn sampleFlow(s: *const Sim, x: f32, z: f32) [2]f32 {
        const gx = x * (1.0 / cell_size);
        const gz = z * (1.0 / cell_size);
        const ix: usize = @min(@as(usize, @intFromFloat(gx)), grid_dim - 1);
        const iz: usize = @min(@as(usize, @intFromFloat(gz)), grid_dim - 1);
        const fx = gx - @as(f32, @floatFromInt(ix));
        const fz = gz - @as(f32, @floatFromInt(iz));
        const n00 = iz * flow_dim + ix;
        const n10 = n00 + 1;
        const n01 = n00 + flow_dim;
        const n11 = n01 + 1;
        const w00 = (1 - fx) * (1 - fz);
        const w10 = fx * (1 - fz);
        const w01 = (1 - fx) * fz;
        const w11 = fx * fz;
        return .{
            s.flow_x[n00] * w00 + s.flow_x[n10] * w10 + s.flow_x[n01] * w01 + s.flow_x[n11] * w11,
            s.flow_z[n00] * w00 + s.flow_z[n10] * w10 + s.flow_z[n01] * w01 + s.flow_z[n11] * w11,
        };
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

fn sharkCell(x: f32, z: f32) usize {
    const cx: usize = @intFromFloat(x * (1.0 / shark_cell_size));
    const cz: usize = @intFromFloat(z * (1.0 / shark_cell_size));
    return cz * shark_grid_dim + cx;
}

/// Ocean currents: the curl of a sum of moving sine waves. Being divergence-free, it moves
/// fish around in swirls without piling them up in sinks.
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

fn wallPush(p: f32) f32 {
    if (p < wall_margin) return (wall_margin - p) / wall_margin;
    if (p > world_size - wall_margin) return -(p - (world_size - wall_margin)) / wall_margin;
    return 0;
}

fn fishTask(s: *Sim, task: usize, begin: usize, end: usize) void {
    const f = s.fish[s.cur];
    const hist = s.hist[task * cell_count ..][0..cell_count];
    @memset(hist, 0);
    const dt = s.dt;
    const steer = @min(1.0, fish_accel * dt);
    const flee_r2 = flee_radius * flee_radius;

    for (begin..end) |i| {
        var x = f.x[i];
        var z = f.z[i];
        var vx = f.vx[i];
        var vz = f.vz[i];

        // Cruise along the current at a roughly constant speed.
        const flow = s.sampleFlow(x, z);
        const flow_len = @sqrt(flow[0] * flow[0] + flow[1] * flow[1]);
        const flow_scale = fish_cruise_speed / @max(flow_len, 0.5);
        var dx = flow[0] * flow_scale + wallPush(x) * fish_cruise_speed * 2;
        var dz = flow[1] * flow_scale + wallPush(z) * fish_cruise_speed * 2;

        // Flee the nearest shark in the 3x3 shark cells around us.
        const scx: usize = @intFromFloat(x * (1.0 / shark_cell_size));
        const scz: usize = @intFromFloat(z * (1.0 / shark_cell_size));
        const x0 = scx -| 1;
        const x1 = @min(scx + 1, shark_grid_dim - 1);
        var best_d2: f32 = flee_r2;
        var away_x: f32 = 0;
        var away_z: f32 = 0;
        for ((scz -| 1)..@min(scz + 1, shark_grid_dim - 1) + 1) |cz| {
            const row = cz * shark_grid_dim;
            // Cells in a row are contiguous, so scan the whole row segment at once.
            for (s.shark_cell_start[row + x0]..s.shark_cell_start[row + x1 + 1]) |k| {
                const ax = x - s.shark_sorted_x[k];
                const az = z - s.shark_sorted_z[k];
                const d2 = ax * ax + az * az;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    away_x = ax;
                    away_z = az;
                }
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

        var phase = f.phase[i] + dt * (3.0 + 2.0 * @sqrt(speed2));
        if (phase > std.math.tau) phase -= std.math.tau;

        f.x[i] = x;
        f.z[i] = z;
        f.vx[i] = vx;
        f.vz[i] = vz;
        f.phase[i] = phase;

        const c = fishCell(x, z);
        s.fish_cell[i] = c;
        hist[c] += 1;
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

fn scatterTask(s: *Sim, task: usize, begin: usize, end: usize) void {
    const src = s.fish[s.cur];
    const dst = s.fish[s.cur ^ 1];
    const cursor = s.hist[task * cell_count ..][0..cell_count];
    for (begin..end) |i| {
        const c = s.fish_cell[i];
        const d = cursor[c];
        cursor[c] = d + 1;
        dst.x[d] = src.x[i];
        dst.z[d] = src.z[i];
        dst.vx[d] = src.vx[i];
        dst.vz[d] = src.vz[i];
        dst.phase[d] = src.phase[i];
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
        var vx = sh.vx[i];
        var vz = sh.vz[i];

        // Nearest fish in the 3x3 fish cells around us.
        const cx: usize = @intFromFloat(x * (1.0 / cell_size));
        const cz: usize = @intFromFloat(z * (1.0 / cell_size));
        const x0 = cx -| 1;
        const x1 = @min(cx + 1, grid_dim - 1);
        var best_d2 = std.math.inf(f32);
        var best: u32 = no_prey;
        for ((cz -| 1)..@min(cz + 1, grid_dim - 1) + 1) |row_z| {
            const row = row_z * grid_dim;
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
            const flow = s.sampleFlow(x, z);
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

        var phase = sh.phase[i] + dt * (2.0 + 0.6 * @sqrt(speed2));
        if (phase > std.math.tau) phase -= std.math.tau;

        sh.x[i] = x;
        sh.z[i] = z;
        sh.vx[i] = vx;
        sh.vz[i] = vz;
        sh.phase[i] = phase;
    }
}
