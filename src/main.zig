const std = @import("std");
const rl = @import("raylib");
const sim = @import("sim.zig");
const render = @import("render.zig");

const min_height: f32 = 6;
const max_height: f32 = 8500;
/// Camera sits behind the target by this fraction of its height, giving a ~70 degree pitch.
const camera_back: f32 = 0.35;
const fovy: f32 = 45;

const ocean = rl.Color{ .r = 12, .g = 44, .b = 74, .a = 255 };
const seabed = rl.Color{ .r = 22, .g = 70, .b = 96, .a = 255 };

const Options = struct {
    /// Render this many frames, save a screenshot, print average timings and exit.
    shot: ?[:0]const u8 = null,
    frames: u32 = 240,
    height: f32 = 60,
    follow: bool = false,
    threads: ?usize = null,
};

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const opts = try parseArgs(try init.minimal.args.toSlice(init.arena.allocator()));

    rl.setConfigFlags(.{ .window_resizable = true, .msaa_4x_hint = true, .vsync_hint = opts.shot == null });
    rl.initWindow(1600, 900, "1 Million Objects");
    defer rl.closeWindow();

    // Leave one hardware thread for the main thread, which uploads and draws while the sim runs.
    const task_count = opts.threads orelse @max(1, (std.Thread.getCpuCount() catch 4) - 1);
    const s = try sim.Sim.create(gpa, init.io, task_count);
    defer s.destroy();
    var renderer = try render.Renderer.init(gpa);

    var target_x: f32 = sim.world_size / 2;
    var target_z: f32 = sim.world_size / 2;
    var height: f32 = opts.height;
    var paused = false;
    var follow: ?usize = null;
    if (opts.follow) follow = nearestShark(s, target_x, target_z);

    var frame: u32 = 0;
    var sum_sim: sim.Timings = .{};
    var sum_cull: f32 = 0;
    var sum_draw: f32 = 0;
    var sum_frame: f32 = 0;

    while (!rl.windowShouldClose()) : (frame += 1) {
        const dt = if (opts.shot != null) 1.0 / 60.0 else @min(rl.getFrameTime(), 1.0 / 30.0);

        // Input
        const pan_speed = height * 1.2 * dt;
        const keys_pan = rl.isKeyDown(.w) or rl.isKeyDown(.a) or rl.isKeyDown(.s) or rl.isKeyDown(.d) or
            rl.isKeyDown(.up) or rl.isKeyDown(.down) or rl.isKeyDown(.left) or rl.isKeyDown(.right);
        if (rl.isKeyDown(.w) or rl.isKeyDown(.up)) target_z -= pan_speed;
        if (rl.isKeyDown(.s) or rl.isKeyDown(.down)) target_z += pan_speed;
        if (rl.isKeyDown(.a) or rl.isKeyDown(.left)) target_x -= pan_speed;
        if (rl.isKeyDown(.d) or rl.isKeyDown(.right)) target_x += pan_speed;
        const dragging = rl.isMouseButtonDown(.left) or rl.isMouseButtonDown(.right);
        if (dragging) {
            const d = rl.getMouseDelta();
            const dist = height * @sqrt(1 + camera_back * camera_back);
            const units_per_px = 2 * dist * @tan(fovy * std.math.pi / 360.0) / @as(f32, @floatFromInt(rl.getScreenHeight()));
            target_x -= d.x * units_per_px;
            target_z -= d.y * units_per_px;
        }
        if (keys_pan or dragging) follow = null;
        height = std.math.clamp(height * std.math.pow(f32, 0.85, rl.getMouseWheelMove()), min_height, max_height);
        if (rl.isKeyPressed(.equal) or rl.isKeyPressed(.kp_add)) height = @max(min_height, height * 0.7);
        if (rl.isKeyPressed(.minus) or rl.isKeyPressed(.kp_subtract)) height = @min(max_height, height / 0.7);
        if (rl.isKeyPressed(.space)) paused = !paused;
        if (rl.isKeyPressed(.f)) follow = if (follow == null) nearestShark(s, target_x, target_z) else null;
        target_x = std.math.clamp(target_x, 0, sim.world_size);
        target_z = std.math.clamp(target_z, 0, sim.world_size);

        if (follow) |i| {
            target_x = s.sharks.x[i];
            target_z = s.sharks.z[i];
        }

        const camera: rl.Camera3D = .{
            .position = .{ .x = target_x, .y = height, .z = target_z + height * camera_back },
            .target = .{ .x = target_x, .y = 0, .z = target_z },
            .up = .{ .x = 0, .y = 1, .z = 0 },
            .fovy = fovy,
            .projection = .perspective,
        };

        // Cull and snapshot the current state, then simulate the next state on another
        // thread while this one uploads and draws. Nothing below may touch `s` until the
        // join except through what `prepare` captured.
        renderer.prepare(s, camera);
        const hud: HudSnapshot = .{ .timings = s.timings, .eaten = s.eaten_total };
        const sim_thread: ?std.Thread = if (paused) null else try std.Thread.spawn(.{}, sim.Sim.step, .{ s, dt });

        rl.beginDrawing();
        rl.clearBackground(ocean);
        // raylib's default far plane (4000) would clip the ocean when zoomed out. Scaling
        // both planes with height also keeps depth precision constant across zoom levels.
        rl.gl.rlSetClipPlanes(height * 0.05, height * 4);
        rl.beginMode3D(camera);
        // The seabed is drawn first and without depth, so it can never z-fight the fish.
        // From high up the depth buffer cannot separate the two and draws bands of seabed.
        rl.gl.rlDisableDepthTest();
        rl.drawPlane(.{ .x = sim.world_size / 2, .y = -3, .z = sim.world_size / 2 }, .{ .x = sim.world_size, .y = sim.world_size }, seabed);
        rl.gl.rlDrawRenderBatchActive();
        rl.gl.rlEnableDepthTest();
        renderer.draw(s);
        rl.endMode3D();
        drawHud(hud, &renderer, height, paused, follow != null);
        rl.endDrawing();

        if (sim_thread) |t| t.join();

        sum_sim.flow += s.timings.flow;
        sum_sim.fish += s.timings.fish;
        sum_sim.sort += s.timings.sort;
        sum_sim.sharks += s.timings.sharks;
        sum_sim.total += s.timings.total;
        sum_cull += renderer.stats.cull_ms;
        sum_draw += renderer.stats.draw_ms;
        sum_frame += rl.getFrameTime() * 1000;

        if (opts.shot) |path| if (frame + 1 == opts.frames) {
            rl.takeScreenshot(path);
            const n: f32 = @floatFromInt(opts.frames);
            const st = renderer.stats;
            std.debug.print(
                "frames={d} threads={d} visible_fish={d} visible_sharks={d} runs={d} tris={d} lod={s}\n" ++
                    "avg ms: frame={d:.2} sim={d:.2} (flow={d:.2} sort={d:.2} fish={d:.2} sharks={d:.2}) cull={d:.2} upload+draw={d:.2}\n",
                .{
                    opts.frames,       task_count,        st.visible_fish,   st.visible_sharks,
                    st.runs,           st.fish_triangles, @tagName(st.lod),  sum_frame / n,
                    sum_sim.total / n, sum_sim.flow / n,  sum_sim.sort / n,  sum_sim.fish / n,
                    sum_sim.sharks / n, sum_cull / n,     sum_draw / n,
                },
            );
            break;
        };
    }
}

fn parseArgs(args: []const [:0]const u8) !Options {
    var o: Options = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--follow")) {
            o.follow = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingArgumentValue;
        i += 1;
        if (std.mem.eql(u8, a, "--shot")) {
            o.shot = args[i];
        } else if (std.mem.eql(u8, a, "--frames")) {
            o.frames = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--threads")) {
            o.threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--height")) {
            o.height = try std.fmt.parseFloat(f32, args[i]);
        } else return error.UnknownArgument;
    }
    return o;
}

fn nearestShark(s: *const sim.Sim, x: f32, z: f32) usize {
    var best: usize = 0;
    var best_d2 = std.math.inf(f32);
    for (0..sim.shark_count) |i| {
        const dx = s.sharks.x[i] - x;
        const dz = s.sharks.z[i] - z;
        if (dx * dx + dz * dz < best_d2) {
            best_d2 = dx * dx + dz * dz;
            best = i;
        }
    }
    return best;
}

/// Simulation values the HUD shows, copied before the step starts running in the background.
const HudSnapshot = struct {
    timings: sim.Timings,
    eaten: u64,
};

fn drawHud(hud: HudSnapshot, r: *const render.Renderer, height: f32, paused: bool, following: bool) void {
    var buf: [256]u8 = undefined;
    const st = r.stats;
    const t = hud.timings;
    rl.drawRectangle(8, 8, 560, 190, rl.fade(.black, 0.55));
    var y: i32 = 16;
    const items = .{
        .{ "{d} FPS  ({d:.2} ms/frame)", .{ rl.getFPS(), rl.getFrameTime() * 1000 } },
        .{ "{d} fish + {d} sharks, {d} eaten", .{ sim.fish_count, sim.shark_count, hud.eaten } },
        .{ "visible: {d} fish, {d} sharks ({d} upload runs)", .{ st.visible_fish, st.visible_sharks, st.runs } },
        .{ "fish mesh: {s}, {d} triangles", .{ @tagName(st.lod), st.fish_triangles } },
        .{ "sim {d:.2} ms: flow {d:.2}  sort {d:.2}  fish {d:.2}  sharks {d:.2}", .{ t.total, t.flow, t.sort, t.fish, t.sharks } },
        .{ "cull {d:.2} ms   upload+draw {d:.2} ms", .{ st.cull_ms, st.draw_ms } },
        .{ "camera height {d:.0}{s}{s}", .{ height, if (paused) "   PAUSED" else "", if (following) "   following shark" else "" } },
        .{ "WASD/drag pan  wheel zoom  F follow shark  Space pause", .{} },
    };
    inline for (items) |item| {
        const text = std.fmt.bufPrintZ(&buf, item[0], item[1]) catch "";
        rl.drawText(text, 18, y, 20, .ray_white);
        y += 22;
    }
}
