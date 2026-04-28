#pragma once
#include "gui/interface/Point.h"
#include "simulation/ElementGraphics.h"
#include "simulation/ElementDefs.h"
#include "FindingElement.h"
#include <cstdint>
#include <optional>
#include <variant>

struct HdispLimitExplicit
{
	float value;
};
struct HdispLimitAuto
{
};
using HdispLimit = std::variant<
	HdispLimitExplicit,
	HdispLimitAuto
>;

struct RendererSettings
{
	uint32_t renderMode = RENDER_BASC | RENDER_FIRE | RENDER_SPRK | RENDER_EFFE;
	uint32_t displayMode = 0;
	uint32_t colorMode = COLOUR_DEFAULT;
	std::optional<FindingElement> findingElement;
	bool gravityZonesEnabled = false;
	bool gravityFieldEnabled = false;
	enum DecorationLevel
	{
		decorationDisabled,
		decorationEnabled,
		decorationAntiClickbait,
	};
	DecorationLevel decorationLevel = decorationEnabled;
	bool debugLines = false;
	ui::Point mousePos = { 0, 0 };
	bool rayTraceEnabled = false;
	int rayTraceResolution = XRES;
	int rayTraceCircleRadius = 64;
	float rayTraceSunIntensity = 1.0f;
	float rayTraceWallIntensity = 0.3f;
	int gridSize = 0;
	float fireIntensity = 1;
	HdispLimit wantHdispLimitMin = HdispLimitExplicit{ MIN_TEMP };
	HdispLimit wantHdispLimitMax = HdispLimitExplicit{ MAX_TEMP };
	Rect<int> autoHdispLimitArea = RES.OriginRect();
};
