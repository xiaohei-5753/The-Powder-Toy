# TPT GPU Ray Traced Lighting Mod — 开发经验总结

## 项目概述

将 [Easy Renderer](https://github.com/)（一个基于 OpenGL Compute Shader 的 2D 光线追踪渲染器）的渲染算法移植到 The Powder Toy (TPT) 中，作为后处理光照效果。

**核心思路**：粒子场景网格化（Material Grid），保留粒子材质信息（元素固有颜色、装饰色、自发光、透明度），然后在 GPU 上用扫描线光线传播算法计算光照，最后合成回 TPT 的帧缓冲区。

---

## 架构

```
Simulation 粒子状态 (parts[])
        │
        ▼
  ┌─────────────────────┐
  │ BuildColorTexture   │  ← 遍历 parts[]，调用 Graphics 函数
  │                     │     计算最终颜色 + 自发光 + 透明度
  │ colorUpload_        │     上传到 cvsColorTex_ (RGBA8)
  │ emissionUpload_     │     上传到 cvsEmissionTex_ (RGB8)
  │ occUpload_          │     上传到 cvsOccuTex_ (R8)
  └──────────┬──────────┘
             │ glTexSubImage2D
             ▼
  ┌─────────────────────┐
  │ Scanline Light      │  ← clearProg_ (清零扫描纹理)
  │ Propagation         │     scanProg_ (512 个方向, 每方向 W+H 条射线)
  │                     │     blendProg_ (混合扫描光 + 像素颜色)
  │ cvsScanTex_ (16F)   │     输出到 renderTex_ (RGBA8)
  └──────────┬──────────┘
             │ PBO 异步读回
             ▼
  ┌─────────────────────┐
  │ ReadBackFromPBO     │  ← 映射 PBO → 缩放 → 写入 g->Data()
  │                     │     (BGR0 格式)
  └──────────┬──────────┘
             │ SDL blit (不变)
             ▼
         屏幕显示
```

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `src/graphics/RayTraceRenderer.h` | `RayTraceParams` 参数结构体 + `RayTraceRenderer` 类 |
| `src/graphics/RayTraceRenderer.cpp` | 全部 GL 实现：context、shader、材质网格、扫描线、读回 |
| `src/graphics/RendererSettings.h` | `rayTraceEnabled` 等设置项，通过 Settings 面板调节 |
| `src/graphics/Renderer.cpp` | TPT 原生渲染器（参考其 `render_parts` 管线） |
| `src/gui/game/GameView.cpp` | `OnDraw()` 中调用 `ProcessFrame()`，`Ctrl+R` 快捷键 |
| `src/gui/options/OptionsView.cpp` | Settings 面板中的光追控制 UI |
| `src/gui/options/OptionsModel.cpp` | Settings 面板的数据模型 |
| `src/gui/options/OptionsController.cpp` | Settings 面板的控制器 |
| `src/PowderToySDL.cpp` | `blit()` + SDL 事件循环（`SDL_QUIT` 修复） |
| `src/graphics/meson.build` | 构建配置（添加 `RayTraceRenderer.cpp`） |
| `meson.build` | GLEW + OpenGL 链接 |
| `third_party/glew/` | GLEW 扩展加载库 |
| `build.bat` | 一键构建脚本 |

---

## 踩坑记录

### 1. 粒子采样方式：pmap → parts[]

**问题**：最初用 `pmap[y][x]` 定位粒子，但 pmap 中粒子 ID 会被复用，且点采样漏掉中间粒子。
**修复**：改为遍历 `parts.data[0..parts.active-1]`，用 `(int)(p.x + 0.5f)` 取像素坐标。
**教训**：TPT 的 pmap 是每帧重建的临时空间哈希，不适合做精确的逐粒子遍历。

### 2. 颜色来源：渲染帧 → 元素固有颜色

**问题**：最初用 `renderedFrame`（已渲染帧的像素）取色，导致粒子移动时颜色错位。
**修复**：改用 `elements[t].Colour`（元素固有颜色）→ 调用 `Graphics` 函数 → 装饰色混叠。
**教训**：粒子已渲染帧的颜色是从上一帧来的，与当前粒子位置不同步。

### 3. Graphics 函数调用

**问题**：不确定如何调用每个元素的自定义 Graphics 函数。
**修复**：直接调用 `elements[t].Graphics(gfctx, &p, nx, ny, &pixel_mode, &cola, &colr, &colg, &colb, &firea, &firer, &fireg, &fireb)`。构造 `GraphicsFuncContext` 时需要 `RendererSettings` 指针和 `RNG`。
**教训**：Graphics 函数会修改颜色、透明度、像素模式、fire 值等输出参数。部分元素返回 0 表示不缓存结果（如 FILT 每个粒子颜色不同）。

### 4. 半透明粒子颜色透传

**问题**：FILT 等半透明粒子的颜色没有混入透过它的光线。
**修复（扫描线版本）**：`light = light * (1 - a) + e + c.rgb * (1 - a)`——不透明像素阻断颜色传播，半透明像素携带颜色。
**修复（逐像素版本）**：`light += trans * max(ref, c.rgb) * c.a`。
**教训**：光线穿过半透明像素时应携带其颜色，但不透明像素不应传播颜色。

### 5. FIRE/NEUT 等纯发光粒子不显示

**问题**：`pixel_mode = PMODE_NONE | FIRE_ADD` 的粒子被判为 alpha=255（不透明），shader 提前返回不处理发光。
**修复**：检测 `!hasPixelMode && hasFireMode` → alpha=0（透明），让 shader 正常处理。
**教训**：`PMODE` 掩码 (`0x00000FFF`) 和 `FIREMODE` 掩码 (`0x00FF0000`) 要分开检查。

### 6. PMODE_GLOW 被 renderMode 过滤

**问题**：`pixel_mode &= settings.renderMode` 过滤掉 `PMODE_GLOW`（默认 renderMode 不含 RENDER_GLOW）。
**修复**：保存 `rawPixelMode`（过滤前的原始值），供 emission 和 alpha 判断使用。
**教训**：renderMode 是用户可配置的显示滤镜，不应影响材质属性。

### 7. 仅渲染左上角

**问题**：低分辨率下（如 306×192）只填了帧缓冲区的左上角。
**修复**：`ReadBack` 中做缩放映射：`rtX = tx * width_ / XRES`, `rtY = ty * height_ / YRES`。
**教训**：CPU 读回时要做坐标变换，不能直接 1:1 拷贝。

### 8. 自发光被计算两次（扫描线版本）

**问题**：`light = light * (1-a) + e` 把 emission 加到 light 里 → `scan[P] += light` 存进 scan → blend 又加一次 `e`。双加导致过亮。
**修复**：只需要一个加法点。最终方案：`light = light * (1-a) + e + c.rgb*(1-a)` → `scan[P] += light` → blend 中 `mix(scan, c.rgb, c.a)` **不加 e**。
**教训**：追踪每个像素的 emission 在整个管线中流经哪些步骤，避免重复叠加。

### 9. 关闭按钮无法退出

**问题**：点击窗口 X 按钮无法关闭游戏。
**修复**：`SDL_QUIT` 事件处理中移除 `engine.CloseWindow()` 调用（它只关闭引擎内部子窗口，不退出），直接 `engine.Exit()`。
**教训**：`SDL_QUIT` 事件只来自主窗口的关闭按钮，不应该走引擎的窗口栈弹出逻辑。

### 10. 扫描线方向竞争导致闪烁

**问题**：移除 direction 间 `glMemoryBarrier` 后，`imageLoad→add→imageStore` 的竞争条件导致帧间闪烁。
**修复**：每 N 个 direction 插一个 barrier（默认 16）。加 `temporalBlend` 帧间混合选项。
**教训**：加法交换律解决不了非原子读写的数据竞争。写后写 (WAW) 危害导致丢失贡献，且丢失的随机性造成闪烁。

---

## 性能优化

| 优化 | 效果 | 说明 |
|------|------|------|
| **扫描线算法** | ~200× texelFetch 减少 | 从逐像素 ~512 条光线改为从边界射入 W+H 条 |
| **异步 PBO 读回** | 消除 CPU 等待 GPU | 双缓冲 PBO，CPU 不阻塞 |
| **GPU 清零** | 消除 1.8MB/帧 CPU→GPU 上传 | compute shader 替代 glTexSubImage2D |
| **批次 Barrier** | 512次 → 32次 | barrier 每 16 个 direction 一次，而非每个 |
| **frameSkip** | 可选降帧渲染 | 每 N 帧才跑一次光追，中间帧复用 PBO |

---

## TPT 渲染管线关键知识点

### 粒子渲染阶段（render_parts）

```
Phase 1: 默认颜色 = elements[t].Colour
Phase 2: 调用 elements[t].Graphics()  → 修改 colr/colg/colb, cola, pixel_mode, fire*
Phase 3: PROP_HOT_GLOW → 高温金属发红（sin 波叠加）
Phase 4: renderMode 过滤 → pixel_mode &= settings.renderMode
Phase 5: ColorMode → COLOUR_HEAT / LIFE / GRAD / BASC
Phase 6: 装饰色混叠 → dcolour alpha blend
Phase 7: 像素绘制 → 根据 pixel_mode 选择绘制方式
```

### pixel_mode 掩码

| 掩码 | 值 | 含义 |
|------|-----|------|
| `PMODE` | `0x00000FFF` | 像素绘制模式 |
| `PMODE_FLAT` | `0x00000001` | 直接写入 |
| `PMODE_BLEND` | `0x00000020` | Alpha 混合 |
| `PMODE_GLOW` | `0x00000008` | 加性发光扩散 |
| `PMODE_BLUR` | `0x00000004` | 模糊扩散 |
| `FIREMODE` | `0x00FF0000` | 火焰模式 |
| `FIRE_ADD` | `0x00010000` | 加性火焰 |
| `FIRE_BLEND` | `0x00020000` | 混合火焰 |

### 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `XRES` | 612 | 模拟区宽 |
| `YRES` | 384 | 模拟区高 |
| `WINDOWW` | 629 | 帧缓冲区宽 (= XRES + BARSIZE) |
| `WINDOWH` | 441 | 帧缓冲区高 (= YRES + MENUSIZE) |
| `CELL` | 4 | 压力/重力/墙壁图块大小 |
| `NPART` | 235,008 | 最大粒子总数 (= XRES * YRES) |

### 数据结构

- `Parts`: 包装 `std::array<Particle, NPART>`，`active` 字段标记活动粒子数
- `Particle`: `{ type, life, ctype, x, y, vx, vy, temp, dcolour, ... }`
- `Element`: `{ Colour, Properties, Graphics, Update, ... }`
- `RenderableSimulation`: `{ pmap, bmap, emap, parts, signs, ... }`
- `RendererSettings`: `{ renderMode, displayMode, colorMode, rayTraceEnabled, ... }`

---

## 构建

```bash
# Debug 构建（双击 build.bat）
build.bat

# Release 构建
meson setup build --buildtype=release
meson compile -C build
```

**依赖**：
- Visual Studio 2022 (MSVC)
- Meson + Ninja
- OpenGL 4.3+ 显卡（Compute Shader 支持）
- GLEW（已包含在 `third_party/glew/`）

**注意**：VS2022 安装在 E 盘时 `--vsenv` 无法自动检测，需先执行 `vcvarsall.bat x64`。
