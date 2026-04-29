#pragma once
#include "Pixel.h"
#include "SimulationConfig.h"
#include "RendererSettings.h"
#include "simulation/Simulation.h"
#include <cstdint>
#include <memory>
#include <vector>

// OpenGL / GLEW forward declarations
struct GLFWwindow;
typedef unsigned int GLuint;
typedef int GLint;

// All tunable parameters in one place for manual adjustment
struct RayTraceParams
{
	// Lighting
	float sunIntensity   = 0.0f;    // disabled (0 = no sun)
	float wallIntensity  = 0.25f;   // ambient = RGB(64,64,64) ≈ 0.25
	float sunSpread      = 0.0f;    // disabled

	// Emission
	float emissiveScale  = 1.0f;

	// Sampling
	int resolution = XRES;
	int circleRadius = 64;

	// Performance: only ray trace every N frames (1 = every frame)
	int frameSkip = 2;

	// Fire glow → emission mapping
	float fireGlowScale  = 1.0f;
};

class RayTraceRenderer
{
public:
	RayTraceRenderer();
	~RayTraceRenderer();

	// Init OpenGL context, compile shaders, create textures
	bool Init();

	// Tear down OpenGL resources
	void Shutdown();

	// Full frame pipeline: consume previous async readback → upload → compute → start next readback.
	// targetStride = width of the target buffer (e.g. WINDOWW).
	void ProcessFrame(pixel *target, int targetStride, int targetHeight,
		const RenderableSimulation &sim, const RendererSettings &settings);

	// Build material grid from simulation state and upload to GPU.
	void UploadMaterialGrid(const RenderableSimulation &sim, const RendererSettings &settings);

	// Dispatch ray tracing compute shader
	void Render();

	// ---- Configuration ----
	void SetEnabled(bool enabled) { enabled_ = enabled; }
	bool IsEnabled() const { return enabled_; }

	void SetParams(const RayTraceParams &p) { params_ = p; }
	const RayTraceParams &GetParams() const { return params_; }
	RayTraceParams &GetParams() { return params_; }

	void SetResolution(int w, int h);
	int GetWidth() const { return width_; }
	int GetHeight() const { return height_; }

private:
	// OpenGL context management
	bool CreateGLContext();
	void DestroyGLContext();

	// Shader management
	bool CompileShaders();
	void DestroyShaders();

	// Texture management
	bool CreateTextures();
	void DestroyTextures();

	// Internal helpers for material grid upload
	void BuildColorTexture(const RenderableSimulation &sim, const RendererSettings &settings);
	void BuildEmissionTexture(const RenderableSimulation &sim, const RendererSettings &settings);
	void BuildOccupancyTexture(const RenderableSimulation &sim);

	// Async PBO readback
	void AsyncReadBack();
	void ReadBackFromPBO(pixel *target, int targetStride, int targetHeight);

	// ---- State ----
	bool enabled_ = false;
	bool initialized_ = false;
	int width_ = XRES;
	int height_ = YRES;
	RayTraceParams params_;

	// OpenGL context
	void *glContext_ = nullptr;   // SDL_GLContext (opaque)
	void *hiddenWindow_ = nullptr; // SDL_Window* (opaque) for GL context

	// OpenGL resource IDs (GLuint)
	GLuint cvsColorTex_ = 0;
	GLuint cvsEmissionTex_ = 0;
	GLuint cvsOccuTex_ = 0;
	GLuint cvsScanTex_ = 0; // float16 accumulation buffer (scanline)
	GLuint renderTex_ = 0;
	GLuint scanProg_ = 0;
	GLuint blendProg_ = 0;

	// Async PBO readback (double buffering)
	static constexpr int kNumPBOs = 2;
	GLuint pbos_[kNumPBOs] = {};
	int curPbo_ = 0;
	bool pboReady_ = false; // true once first async readback has completed
	int frameCounter_ = 0;  // for frame-skip throttling

	// Temp buffers for upload
	std::vector<uint8_t> colorUpload_;
	std::vector<uint8_t> emissionUpload_;
	std::vector<uint8_t> occUpload_;
};
