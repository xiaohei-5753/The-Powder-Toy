#include "RayTraceRenderer.h"
#include "Config.h"
#include "common/platform/Platform.h"
#include "graphics/Renderer.h"
#include "simulation/ElementDefs.h"
#include "simulation/ElementGraphics.h"
#include "simulation/SimulationData.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <GL/glew.h>
#include <SDL.h>

// ============================================================================
// Compute shader — ported from easy_renderer
// ============================================================================

static const char *rayCompSrc = R"(#version 430 core
layout(local_size_x = 8, local_size_y = 8) in;

uniform sampler2D u_cc;
uniform sampler2D u_cl;
uniform sampler2D u_occ;
uniform vec2  u_sd;
uniform float u_ss;
uniform vec3  u_sl;
uniform vec3  u_wl;
uniform int   u_cr;
uniform ivec2 u_cs;
uniform float u_ep;

layout(rgba8, binding = 0) writeonly uniform image2D u_out;

bool isEmptyRegion(ivec2 p) {
	float occ = texelFetch(u_occ, p, 0).r;
	return occ < 0.5;
}

vec3 cR(ivec2 o, ivec2 d) {
	vec3 light = vec3(0.0);
	float trans = 1.0;
	int dx = abs(d.x), dy = abs(d.y);
	int sx = (d.x >= 0) ? 1 : -1, sy = (d.y >= 0) ? 1 : -1;
	int er = dx - dy, x = o.x, y = o.y;
	int e2 = 2 * er;
	if (e2 > -dy) { er -= dy; x += sx; }
	if (e2 <  dx) { er += dx; y += sy; }
	while (true) {
		if (x < 0 || x >= u_cs.x || y < 0 || y >= u_cs.y) {
			vec2 dn = normalize(vec2(d));
			if (1.0 - dot(dn, u_sd) <= u_ss)
				light += trans * u_sl;
			else
				light += trans * u_wl;
			return light;
		}
		if (isEmptyRegion(ivec2(x, y))) {
			for (int s = 0; s < 4; s++) {
				e2 = 2 * er;
				if (e2 > -dy) { er -= dy; x += sx; }
				if (e2 <  dx) { er += dx; y += sy; }
				if (x < 0 || x >= u_cs.x || y < 0 || y >= u_cs.y) break;
				if (!isEmptyRegion(ivec2(x, y))) break;
			}
			continue;
		}
		vec4 c = texelFetch(u_cc, ivec2(x, y), 0);
		vec3 ref = texelFetch(u_cl, ivec2(x, y), 0).rgb;
		if (c.a >= 1.0) {
			light += trans * ref;
			return light;
		}
		if (c.a > 0.0) {
			// Add both emission AND pixel colour to the passing light.
			// This makes semi-transparent particles (like FILT) tint
			// the light that passes through them.
			light += trans * max(ref, c.rgb) * c.a;
			trans *= (1.0 - c.a);
			if (trans < u_ep) return light;
		}
		e2 = 2 * er;
		if (e2 > -dy) { er -= dy; x += sx; }
		if (e2 <  dx) { er += dx; y += sy; }
	}
}

void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (p.x >= u_cs.x || p.y >= u_cs.y) return;

	vec4 c = texelFetch(u_cc, p, 0);
	if (c.a >= 1.0) {
		imageStore(u_out, p, c);
		return;
	}
	vec3 a = vec3(0.0);
	int cnt = 0, r = u_cr, m = 1 - r, x = 0, y = r;
	while (x < y) {
		a += cR(p, ivec2( x,  y)); cnt++;
		a += cR(p, ivec2(-x,  y)); cnt++;
		a += cR(p, ivec2( x, -y)); cnt++;
		a += cR(p, ivec2(-x, -y)); cnt++;
		a += cR(p, ivec2( y,  x)); cnt++;
		a += cR(p, ivec2(-y,  x)); cnt++;
		a += cR(p, ivec2( y, -x)); cnt++;
		a += cR(p, ivec2(-y, -x)); cnt++;
		x++;
		if (m < 0)
			m += 2 * x + 1;
		else {
			y--;
			m += 2 * (x - y) + 1;
		}
	}
	a /= float(cnt);
	vec3 selfLight = texelFetch(u_cl, p, 0).rgb;
	a += selfLight;
	a = max(a, vec3(0.0));
	float ov = max(max(a.r, a.g), a.b);
	if (ov > 1.0) a /= ov;
	imageStore(u_out, p, vec4(mix(a, c.rgb, c.a), 1.0));
}
)";

static const char *dispVertSrc = R"(#version 330 core
layout(location = 0) in vec2 aP;
layout(location = 1) in vec2 aU;
out vec2 vU;
void main() {
	gl_Position = vec4(aP, 0.0, 1.0);
	vU = aU;
}
)";

static const char *dispFragSrc = R"(#version 330 core
uniform sampler2D u_t;
in vec2 vU;
out vec4 oC;
void main() {
	oC = texture(u_t, vU);
}
)";

// ============================================================================
// RayTraceRenderer Implementation
// ============================================================================

RayTraceRenderer::RayTraceRenderer()
{
}

RayTraceRenderer::~RayTraceRenderer()
{
	Shutdown();
}

bool RayTraceRenderer::Init()
{
	if (initialized_)
		return true;

	if (!CreateGLContext())
		return false;
	if (!CompileShaders())
		return false;
	if (!CreateTextures())
		return false;

	initialized_ = true;
	return true;
}

void RayTraceRenderer::Shutdown()
{
	DestroyTextures();
	DestroyShaders();
	DestroyGLContext();
	initialized_ = false;
}

// ============================================================================
// OpenGL Context Management
// ============================================================================

bool RayTraceRenderer::CreateGLContext()
{
	hiddenWindow_ = SDL_CreateWindow(
		"", 0, 0, 1, 1,
		SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN
	);
	if (!hiddenWindow_)
	{
		fprintf(stderr, "RayTraceRenderer: SDL_CreateWindow (GL) failed: %s\n", SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	glContext_ = SDL_GL_CreateContext(static_cast<SDL_Window *>(hiddenWindow_));
	if (!glContext_)
	{
		fprintf(stderr, "RayTraceRenderer: SDL_GL_CreateContext failed: %s\n", SDL_GetError());
		SDL_DestroyWindow(static_cast<SDL_Window *>(hiddenWindow_));
		hiddenWindow_ = nullptr;
		return false;
	}

	glewExperimental = GL_TRUE;
	GLenum glewErr = glewInit();
	if (glewErr != GLEW_OK)
	{
		fprintf(stderr, "RayTraceRenderer: glewInit failed: %s\n", glewGetErrorString(glewErr));
		SDL_GL_DeleteContext(static_cast<SDL_GLContext>(glContext_));
		glContext_ = nullptr;
		SDL_DestroyWindow(static_cast<SDL_Window *>(hiddenWindow_));
		hiddenWindow_ = nullptr;
		return false;
	}

	if (!GLEW_VERSION_4_3)
	{
		fprintf(stderr, "RayTraceRenderer: OpenGL 4.3 not supported by your driver\n");
		return false;
	}

	fprintf(stdout, "RayTraceRenderer: OpenGL %s initialized (vendor: %s)\n",
		glGetString(GL_VERSION), glGetString(GL_VENDOR));

	return true;
}

void RayTraceRenderer::DestroyGLContext()
{
	if (glContext_)
	{
		SDL_GL_DeleteContext(static_cast<SDL_GLContext>(glContext_));
		glContext_ = nullptr;
	}
	if (hiddenWindow_)
	{
		SDL_DestroyWindow(static_cast<SDL_Window *>(hiddenWindow_));
		hiddenWindow_ = nullptr;
	}
}

// ============================================================================
// Shader Compilation
// ============================================================================

static GLuint CompileGLShader(const char *src, GLenum type)
{
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok)
	{
		char log[4096];
		GLsizei len = 0;
		glGetShaderInfoLog(s, sizeof(log), &len, log);
		fprintf(stderr, "RayTraceRenderer: shader compile error:\n%s\n", log);
	}
	return s;
}

static GLuint LinkGLProgram(GLuint vs, GLuint fs)
{
	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok)
	{
		char log[4096];
		GLsizei len = 0;
		glGetProgramInfoLog(p, sizeof(log), &len, log);
		fprintf(stderr, "RayTraceRenderer: program link error:\n%s\n", log);
	}
	return p;
}

bool RayTraceRenderer::CompileShaders()
{
	GLuint cs = CompileGLShader(rayCompSrc, GL_COMPUTE_SHADER);
	if (!cs) return false;
	rayProg_ = glCreateProgram();
	glAttachShader(rayProg_, cs);
	glLinkProgram(rayProg_);
	{
		GLint ok = 0;
		glGetProgramiv(rayProg_, GL_LINK_STATUS, &ok);
		if (!ok)
		{
			char log[4096];
			GLsizei len = 0;
			glGetProgramInfoLog(rayProg_, sizeof(log), &len, log);
			fprintf(stderr, "RayTraceRenderer: compute shader link error:\n%s\n", log);
			return false;
		}
	}
	glDeleteShader(cs);

	GLuint dvs = CompileGLShader(dispVertSrc, GL_VERTEX_SHADER);
	GLuint dfs = CompileGLShader(dispFragSrc, GL_FRAGMENT_SHADER);
	if (!dvs || !dfs) return false;
	dispProg_ = LinkGLProgram(dvs, dfs);
	glDeleteShader(dvs);
	glDeleteShader(dfs);

	return true;
}

void RayTraceRenderer::DestroyShaders()
{
	if (rayProg_) { glDeleteProgram(rayProg_); rayProg_ = 0; }
	if (dispProg_) { glDeleteProgram(dispProg_); dispProg_ = 0; }
}

// ============================================================================
// Texture Management
// ============================================================================

bool RayTraceRenderer::CreateTextures()
{
	auto createTex = [](GLenum internalFormat, GLenum format, GLenum type, int w, int h) -> GLuint {
		GLuint tex;
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexImage2D(GL_TEXTURE_2D, 0, (GLint)internalFormat, w, h, 0, format, type, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		return tex;
	};

	cvsColorTex_    = createTex(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, width_, height_);
	cvsEmissionTex_ = createTex(GL_RGB8,  GL_RGB,  GL_UNSIGNED_BYTE, width_, height_);
	cvsOccuTex_     = createTex(GL_R8,    GL_RED,  GL_UNSIGNED_BYTE, width_, height_);
	renderTex_      = createTex(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, width_, height_);

	colorUpload_.resize((size_t)width_ * height_ * 4);
	emissionUpload_.resize((size_t)width_ * height_ * 3);
	occUpload_.resize((size_t)width_ * height_);

	float verts[] = { -1, -1, 0, 0, 3, -1, 2, 0, -1, 3, 0, 2 };
	glGenVertexArrays(1, &fullVAO_);
	glBindVertexArray(fullVAO_);
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void *)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void *)8);
	glBindVertexArray(0);

	// Create PBOs for async readback
	size_t pboSize = (size_t)width_ * height_ * 4;
	glGenBuffers(kNumPBOs, pbos_);
	for (int i = 0; i < kNumPBOs; i++)
	{
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[i]);
		glBufferData(GL_PIXEL_PACK_BUFFER, pboSize, nullptr, GL_STREAM_READ);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	return true;
}

void RayTraceRenderer::DestroyTextures()
{
	auto delTex = [](GLuint &t) { if (t) { glDeleteTextures(1, &t); t = 0; } };
	delTex(cvsColorTex_);
	delTex(cvsEmissionTex_);
	delTex(cvsOccuTex_);
	delTex(renderTex_);
	if (fullVAO_) { glDeleteVertexArrays(1, &fullVAO_); fullVAO_ = 0; }
	if (pbos_[0]) { glDeleteBuffers(kNumPBOs, pbos_); pbos_[0] = 0; }
	curPbo_ = 0;
	pboReady_ = false;
}

// ============================================================================
// Resolution
// ============================================================================

void RayTraceRenderer::SetResolution(int w, int h)
{
	if (w == width_ && h == height_)
		return;
	width_ = w;
	height_ = h;
	if (initialized_)
	{
		DestroyTextures();
		CreateTextures();
	}
}

// ============================================================================
// Material Grid Construction
// ============================================================================

void RayTraceRenderer::UploadMaterialGrid(
	const RenderableSimulation &sim,
	const RendererSettings &settings
) {
	BuildColorTexture(sim, settings); // fills both colorUpload_ AND emissionUpload_
	BuildEmissionTexture(sim, settings); // uploads emissionUpload_ to GPU
	BuildOccupancyTexture(sim);
}

// ============================================================================
// Full render_parts color pipeline for the material grid
// ============================================================================

void RayTraceRenderer::BuildColorTexture(
	const RenderableSimulation &sim,
	const RendererSettings &settings
) {
	auto &sd = SimulationData::CRef();
	auto &elements = sd.elements;

	// Clear to transparent; clear emission too (avoids stale data from last frame)
	std::fill(colorUpload_.begin(), colorUpload_.end(), 0);
	std::fill(emissionUpload_.begin(), emissionUpload_.end(), 0);

	// Construct a GraphicsFuncContext for calling element graphics functions
	GraphicsFuncContext gfctx;
	gfctx.ren = &settings;
	gfctx.sim = &sim;
	gfctx.pipeSubcallCpart = nullptr;
	gfctx.pipeSubcallTpart = nullptr;

	for (int i = 0; i < sim.parts.active; i++)
	{
		const auto &p = sim.parts.data[i];
		int t = p.type;
		if (t <= 0 || t >= PT_NUM)
			continue;

		int nx = (int)(p.x + 0.5f);
		int ny = (int)(p.y + 0.5f);
		if (nx < 0 || nx >= XRES || ny < 0 || ny >= YRES)
			continue;

		// === Phase 1: Defaults (mirrors render_parts lines 281-288) ===
		int pixel_mode = PMODE_FLAT;
		int cola = 255;
		RGB colour = elements[t].Colour;
		int colr = colour.Red;
		int colg = colour.Green;
		int colb = colour.Blue;
		int firea = 0, firer = 0, fireg = 0, fireb = 0;

		// Extract decoration
		unsigned int dcolour = p.dcolour;
		int deca = (dcolour >> 24) & 0xFF;
		int decr = (dcolour >> 16) & 0xFF;
		int decg = (dcolour >> 8) & 0xFF;
		int decb = dcolour & 0xFF;

		// Anti-clickbait filter (mirrors render_parts lines 295-304)
		if (settings.decorationLevel == RendererSettings::decorationAntiClickbait)
		{
			if (deca < 250 || decr > 5 || decg > 5 || decb > 5)
				deca = 0;
			else
			{
				deca = 255;
				decr = decg = decb = 0;
			}
		}

		// === Phase 2: Graphics function (mirrors render_parts lines 306-338) ===
		if (!(settings.colorMode & COLOUR_BASC))
		{
			auto *graphics = elements[t].Graphics;
			if (graphics)
			{
				graphics(gfctx, &p, nx, ny, &pixel_mode, &cola, &colr, &colg, &colb, &firea, &firer, &fireg, &fireb);
			}
		}

		// === Phase 3: PROP_HOT_GLOW (mirrors render_parts lines 339-346) ===
		if ((elements[t].Properties & PROP_HOT_GLOW) && p.temp > (elements[t].HighTemperature - 800.0f))
		{
			auto gradv = 3.14159265358979323846 / (2 * elements[t].HighTemperature - (elements[t].HighTemperature - 800.0f));
			auto caddress = int((p.temp > elements[t].HighTemperature)
				? elements[t].HighTemperature - (elements[t].HighTemperature - 800.0f)
				: p.temp - (elements[t].HighTemperature - 800.0f));
			colr += int(sin(gradv * caddress) * 226);
			colg += int(-sin(gradv * caddress * 4.55f) * 34);
			colb += int(-sin(gradv * caddress * 2.22f) * 64);
		}

		// === Phase 4: Save raw pixel_mode for emission/alpha decisions,
		// then apply render mode filtering (mirrors render_parts lines 348-359).
		int rawPixelMode = pixel_mode;

		if ((pixel_mode & FIRE_ADD) && !(settings.renderMode & FIRE_ADD))
			pixel_mode |= PMODE_GLOW;
		if ((pixel_mode & FIRE_BLEND) && !(settings.renderMode & FIRE_BLEND))
			pixel_mode |= PMODE_BLUR;
		if ((pixel_mode & PMODE_BLUR) && !(settings.renderMode & PMODE_BLUR))
			pixel_mode |= PMODE_FLAT;
		if ((pixel_mode & PMODE_GLOW) && !(settings.renderMode & PMODE_GLOW))
			pixel_mode |= PMODE_BLEND;
		if (settings.renderMode & PMODE_BLOB)
			pixel_mode |= PMODE_BLOB;
		pixel_mode &= settings.renderMode;

		// === Phase 5: Color mode (mirrors render_parts lines 361-399) ===
		if (settings.colorMode & COLOUR_HEAT)
		{
			RGB hc = Renderer::heatTableAt(int((p.temp - 0) / (MAX_TEMP - 0) * 1024));
			colr = hc.Red; colg = hc.Green; colb = hc.Blue;
			cola = 255;
			if (pixel_mode & (FIREMODE | PMODE_GLOW))
				pixel_mode = (pixel_mode & ~(FIREMODE | PMODE_GLOW)) | PMODE_BLUR;
			else if ((pixel_mode & (PMODE_BLEND | PMODE_ADD)) == (PMODE_BLEND | PMODE_ADD))
				pixel_mode = (pixel_mode & ~(PMODE_BLEND | PMODE_ADD)) | PMODE_FLAT;
			else if (!pixel_mode)
				pixel_mode |= PMODE_FLAT;
		}
		else if (settings.colorMode & COLOUR_BASC)
		{
			colr = colour.Red; colg = colour.Green; colb = colour.Blue;
			pixel_mode = PMODE_FLAT;
		}
		else if (settings.colorMode & COLOUR_LIFE)
		{
			auto gradv = 0.4f;
			int q = p.life;
			if (q < 5) q = p.life;
			else q = int(sqrt((float)p.life));
			colr = colg = colb = int(sin(gradv * q) * 100 + 128);
			cola = 255;
		}

		// === Phase 6: Decoration blending (mirrors render_parts lines 401-418) ===
		if (!(settings.colorMode & ~COLOUR_GRAD) &&
			settings.decorationLevel != RendererSettings::decorationDisabled && deca)
		{
			deca++;
			if (!(pixel_mode & NO_DECO))
			{
				colr = (deca * decr + (256 - deca) * colr) >> 8;
				colg = (deca * decg + (256 - deca) * colg) >> 8;
				colb = (deca * decb + (256 - deca) * colb) >> 8;
			}
			if (pixel_mode & DECO_FIRE)
			{
				firer = (deca * decr + (256 - deca) * firer) >> 8;
				fireg = (deca * decg + (256 - deca) * fireg) >> 8;
				fireb = (deca * decb + (256 - deca) * fireb) >> 8;
			}
		}

		// COLOUR_GRAD (mirrors render_parts lines 420-428)
		if (settings.colorMode & COLOUR_GRAD)
		{
			auto frequency = 0.05f;
			auto q = int(p.temp - 40);
			colr = int(sin(frequency * q) * 16 + colr);
			colg = int(sin(frequency * q) * 16 + colg);
			colb = int(sin(frequency * q) * 16 + colb);
			if (pixel_mode & (FIREMODE | PMODE_GLOW))
				pixel_mode = (pixel_mode & ~(FIREMODE | PMODE_GLOW)) | PMODE_BLUR;
		}

		// === Clamp (mirrors render_parts lines 430-447) ===
		auto clamp8 = [](int &v) { if (v > 255) v = 255; else if (v < 0) v = 0; };
		clamp8(colr); clamp8(colg); clamp8(colb); clamp8(cola);
		clamp8(firer); clamp8(fireg); clamp8(fireb); clamp8(firea);

		// Map to raytrace pixel
		int rtX = nx * width_ / XRES;
		int rtY = ny * height_ / YRES;
		int idx = rtY * width_ + rtX;

		// === Determine alpha for the material grid ===
		// Use rawPixelMode (before renderMode filtering) for emission/alpha decisions,
		// so particles like BRAY (PMODE_BLEND|PMODE_GLOW) still get their glow
		// even when RENDER_GLOW is disabled.
		bool hasPixelMode = (rawPixelMode & PMODE) != 0;
		bool hasFireMode  = (rawPixelMode & FIREMODE) != 0;
		bool hasGlowMode  = (rawPixelMode & (PMODE_GLOW | PMODE_BLUR | PMODE_ADD | PMODE_SPARK | PMODE_FLARE | PMODE_LFLARE)) != 0;

		int outAlpha;
		if (!hasPixelMode && hasFireMode)
		{
			// Effect-only: FIRE, NEUT — no pixel, only emission
			outAlpha = 0;
		}
		else if (rawPixelMode & PMODE_BLEND)
		{
			outAlpha = cola;
		}
		else if (hasGlowMode)
		{
			// Glow/blur effects: semi-transparent so ray tracer processes them
			outAlpha = cola > 200 ? 180 : cola;
		}
		else
		{
			outAlpha = 255; // PMODE_FLAT, PMODE_BLOB — solid
		}

		// === Write color ===
		colorUpload_[idx * 4 + 0] = (uint8_t)colr;
		colorUpload_[idx * 4 + 1] = (uint8_t)colg;
		colorUpload_[idx * 4 + 2] = (uint8_t)colb;
		colorUpload_[idx * 4 + 3] = (uint8_t)outAlpha;

		// === Compute emission ===
		float er = 0.0f, eg = 0.0f, eb = 0.0f;
		float emissiveIntensity = 0.0f;

		// 1. Emission from fire values (Graphics function set firer/fireg/fireb/firea)
		bool fireModeRaw = (rawPixelMode & FIREMODE) != 0;
		bool glowModeRaw = (rawPixelMode & (PMODE_GLOW | PMODE_BLUR | PMODE_ADD | PMODE_SPARK | PMODE_FLARE | PMODE_LFLARE)) != 0;
		if (fireModeRaw && firea > 0)
		{
			emissiveIntensity = std::max(emissiveIntensity, firea / 255.0f);
			er = std::max(er, firer / 255.0f);
			eg = std::max(eg, fireg / 255.0f);
			eb = std::max(eb, fireb / 255.0f);
		}

		// 2. Emission from temperature (hot solids like metal with PROP_HOT_GLOW)
		if (p.temp > 500.0f && hasPixelMode)
		{
			float tempIntensity = std::min(1.0f, (p.temp - 500.0f) / 2500.0f);
			emissiveIntensity = std::max(emissiveIntensity, tempIntensity);
			er = std::max(er, colr / 255.0f);
			eg = std::max(eg, colg / 255.0f);
			eb = std::max(eb, colb / 255.0f);
		}

		// 3. Always allow some emission from firea if set, even without fire mode
		if (!fireModeRaw && firea > 0 && emissiveIntensity < 0.01f)
		{
			emissiveIntensity = std::max(emissiveIntensity, firea / 255.0f);
			er = std::max(er, firer / 255.0f);
			eg = std::max(eg, fireg / 255.0f);
			eb = std::max(eb, fireb / 255.0f);
		}

		// 4. Emission from fancy pixel modes: GLOW, BLUR, SPARK, FLARE, LFLARE
		// (BRAY, and other "fancy mode" particles that visually glow)
		if (glowModeRaw && emissiveIntensity < 0.01f)
		{
			emissiveIntensity = 0.5f; // moderate default glow
			er = std::max(er, colr / 255.0f);
			eg = std::max(eg, colg / 255.0f);
			eb = std::max(eb, colb / 255.0f);
		}

		emissiveIntensity *= params_.emissiveScale;
		er *= emissiveIntensity;
		eg *= emissiveIntensity;
		eb *= emissiveIntensity;

		emissionUpload_[idx * 3 + 0] = (uint8_t)(std::min(255.0f, er * 255.0f));
		emissionUpload_[idx * 3 + 1] = (uint8_t)(std::min(255.0f, eg * 255.0f));
		emissionUpload_[idx * 3 + 2] = (uint8_t)(std::min(255.0f, eb * 255.0f));
	}

	// Fill walls (opaque grey)
	for (int cy = 0; cy < YCELLS; cy++)
	{
		for (int cx = 0; cx < XCELLS; cx++)
		{
			if (sim.bmap[cy][cx])
			{
				for (int dy = 0; dy < CELL; dy++)
				{
					int py = cy * CELL + dy;
					if (py >= YRES) continue;
					int rtY = py * height_ / YRES;
					for (int dx = 0; dx < CELL; dx++)
					{
						int px = cx * CELL + dx;
						if (px >= XRES) continue;
						int rtX = px * width_ / XRES;
						int idx = rtY * width_ + rtX;
						if (!colorUpload_[idx * 4 + 3])
						{
							colorUpload_[idx * 4 + 0] = 128;
							colorUpload_[idx * 4 + 1] = 128;
							colorUpload_[idx * 4 + 2] = 128;
							colorUpload_[idx * 4 + 3] = 255;
						}
					}
				}
			}
		}
	}

	glBindTexture(GL_TEXTURE_2D, cvsColorTex_);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, colorUpload_.data());
}

// ============================================================================
// Build emission texture from fire values, glow modes, and temperature
// ============================================================================

void RayTraceRenderer::BuildEmissionTexture(
	const RenderableSimulation & /*sim*/,
	const RendererSettings & /*settings*/
) {
	// Emission is now computed inline in BuildColorTexture,
	// using the processed pixel color (colr/colg/colb) modulated
	// by temperature intensity. The emissionUpload_ buffer is already
	// filled; just upload it to the GPU.
	glBindTexture(GL_TEXTURE_2D, cvsEmissionTex_);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, emissionUpload_.data());
}

// ============================================================================
// Build occupancy texture
// ============================================================================

void RayTraceRenderer::BuildOccupancyTexture(const RenderableSimulation &sim)
{
	std::fill(occUpload_.begin(), occUpload_.end(), 0);

	for (int i = 0; i < sim.parts.active; i++)
	{
		const auto &p = sim.parts.data[i];
		if (p.type <= 0 || p.type >= PT_NUM)
			continue;
		int px = (int)(p.x + 0.5f);
		int py = (int)(p.y + 0.5f);
		if (px < 0 || px >= XRES || py < 0 || py >= YRES)
			continue;
		int rtX = px * width_ / XRES;
		int rtY = py * height_ / YRES;
		occUpload_[rtY * width_ + rtX] = 255;
	}

	for (int cy = 0; cy < YCELLS; cy++)
	{
		for (int cx = 0; cx < XCELLS; cx++)
		{
			if (sim.bmap[cy][cx])
			{
				for (int dy = 0; dy < CELL; dy++)
				{
					int py = cy * CELL + dy;
					if (py >= YRES) continue;
					int rtY = py * height_ / YRES;
					for (int dx = 0; dx < CELL; dx++)
					{
						int px = cx * CELL + dx;
						if (px >= XRES) continue;
						int rtX = px * width_ / XRES;
						occUpload_[rtY * width_ + rtX] = 255;
					}
				}
			}
		}
	}

	glBindTexture(GL_TEXTURE_2D, cvsOccuTex_);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RED, GL_UNSIGNED_BYTE, occUpload_.data());
}

// ============================================================================
// Ray Tracing Dispatch
// ============================================================================

void RayTraceRenderer::Render()
{
	if (!initialized_ || !enabled_)
		return;

	float sunDirX = 0.7071f;
	float sunDirY = 0.7071f;
	float sunSc = params_.sunSpread;
	float eps = 1e-3f;

	float sl[3] = {
		params_.sunIntensity * 10.0f,
		params_.sunIntensity * 9.0f,
		params_.sunIntensity * 7.0f,
	};
	float wl[3] = {
		params_.wallIntensity * 0.251f,
		params_.wallIntensity * 0.251f,
		params_.wallIntensity * 0.251f,
	};

	glBindImageTexture(0, renderTex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
	glUseProgram(rayProg_);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, cvsColorTex_);
	glUniform1i(glGetUniformLocation(rayProg_, "u_cc"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, cvsEmissionTex_);
	glUniform1i(glGetUniformLocation(rayProg_, "u_cl"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, cvsOccuTex_);
	glUniform1i(glGetUniformLocation(rayProg_, "u_occ"), 2);

	glUniform2f(glGetUniformLocation(rayProg_, "u_sd"), sunDirX, sunDirY);
	glUniform1f(glGetUniformLocation(rayProg_, "u_ss"), sunSc);
	glUniform3fv(glGetUniformLocation(rayProg_, "u_sl"), 1, sl);
	glUniform3fv(glGetUniformLocation(rayProg_, "u_wl"), 1, wl);
	glUniform1i(glGetUniformLocation(rayProg_, "u_cr"), params_.circleRadius);
	glUniform2i(glGetUniformLocation(rayProg_, "u_cs"), width_, height_);
	glUniform1f(glGetUniformLocation(rayProg_, "u_ep"), eps);

	glDispatchCompute((width_ + 7) / 8, (height_ + 7) / 8, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

// ============================================================================
// ProcessFrame — full pipeline with async PBO readback
// ============================================================================

void RayTraceRenderer::ProcessFrame(pixel *target, int targetStride, int targetHeight,
	const RenderableSimulation &sim, const RendererSettings &settings)
{
	if (!initialized_ || !enabled_) return;

	frameCounter_++;

	// Step 1: consume the PREVIOUS frame's result from PBO (non-blocking)
	// This always runs, even on skipped compute frames, so the display keeps updating
	// with the last good result.
	if (pboReady_)
	{
		ReadBackFromPBO(target, targetStride, targetHeight);
	}

	// Skip compute + upload on throttled frames to let physics run freely
	if (params_.frameSkip > 1 && (frameCounter_ % params_.frameSkip) != 0)
	{
		return;
	}

	// Step 2: upload current frame's material grid
	UploadMaterialGrid(sim, settings);

	// Step 3: dispatch compute shader for current frame
	Render();

	// Step 4: start async readback of current frame into the NEXT PBO
	AsyncReadBack();
}

// ============================================================================
// Async PBO Readback
// ============================================================================

void RayTraceRenderer::AsyncReadBack()
{
	int nextPbo = (curPbo_ + 1) % kNumPBOs;
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[nextPbo]);
	glBindTexture(GL_TEXTURE_2D, renderTex_);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	// GPU writes asynchronously to PBO; CPU returns immediately.
	// The PBO will be consumed on the next frame.
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	curPbo_ = nextPbo;
	pboReady_ = true;
}

void RayTraceRenderer::ReadBackFromPBO(pixel *target, int targetStride, int targetHeight)
{
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos_[curPbo_]);
	const void *mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	if (!mapped)
	{
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		return;
	}
	const uint8_t *pixels = static_cast<const uint8_t *>(mapped);

	// Scale raytrace output to fill the full simulation area (XRES × YRES)
	int outH = std::min(YRES, targetHeight);
	int outW = std::min(XRES, targetStride);
	for (int ty = 0; ty < outH; ty++)
	{
		int rtY = ty * height_ / YRES;
		for (int tx = 0; tx < outW; tx++)
		{
			int rtX = tx * width_ / XRES;
			int srcIdx = (rtY * width_ + rtX) * 4;
			uint8_t r = pixels[srcIdx + 0];
			uint8_t g = pixels[srcIdx + 1];
			uint8_t b = pixels[srcIdx + 2];
			target[ty * targetStride + tx] = RGB(r, g, b).Pack();
		}
	}

	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}
