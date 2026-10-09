# 1MillionObjects
This is a challenge project to show I can render 1 million unique 3d objects

1,000 sharks chase 999,000 fish around a 2000 × 2000 ocean. Fish drift on swirling currents and flee any shark within 20 units. Sharks are faster, and an eaten fish respawns somewhere else, so the count stays at one million.

## Running

Requires Zig 0.16.0. raylib-zig is pinned to the last commit that supports 0.16.

```sh
zig build run -Doptimize=ReleaseFast
```

Debug builds work but are far too slow for a million objects.

| Input | Action |
| --- | --- |
| WASD / arrows, or drag with the mouse | Pan |
| Mouse wheel, `+` / `-` | Zoom |
| `F` | Follow the nearest shark |
| `Space` | Pause |

Benchmark mode renders a fixed number of frames, saves a screenshot and prints average timings:

```sh
zig-out/bin/1MillionObjects --shot out.png --height 2600 --frames 240 [--threads N] [--follow]
```

## How it works

- **Structs of arrays.** Each species is a set of `x`, `z`, `vx`, `vz` and `phase` arrays (`Bodies` in [src/sim.zig](src/sim.zig)).
- **Spatial grid via counting sort.** Every frame the fish are re-sorted into a 125 × 125 grid of 16-unit cells. The sort is parallel and stable (per-thread histograms, then a scatter into a second set of arrays). Each grid row is then one contiguous range of memory, so shark prey searches and view culling read runs of memory instead of chasing indices. Sharks get their own coarser grid, which fish use to find the nearest threat.
- **Threads.** All per-entity work is split into one chunk per core and run on the `std.Io` thread pool.
- **Culling.** The camera frustum is projected onto the ground plane. Only the grid rows and column spans inside it are copied into the instance buffer, and nothing outside it reaches the GPU. Every object is still simulated.
- **Draw calls.** There is one instanced draw for fish and one for sharks. Each instance is 20 bytes (position, velocity, swim phase) instead of a 64-byte matrix. The vertex shader works out the heading from the velocity and animates the swim wiggle. The fish model's three meshes are merged into one mesh with vertex colours, and above a camera height of 140 a 10-triangle low-poly fish replaces the 686-triangle model.

## Credits

3D models are not covered by this project's MIT license. They belong to their creators:

- `shark.glb`: [Poly by Google](https://poly.pizza/u/Poly%20by%20Google), licensed [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- `fish.glb`: [Quaternius](https://quaternius.com/), licensed [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
