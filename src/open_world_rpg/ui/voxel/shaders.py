"""Bundled GLSL for lit, fogged voxel terrain."""

VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec2 in_uv;
in float in_shade;
uniform mat4 projection;
uniform mat4 view;
out vec2 uv;
out float shade;
out float distance_to_camera;
void main() {
    vec4 view_position = view * vec4(in_position, 1.0);
    gl_Position = projection * view_position;
    distance_to_camera = length(view_position.xyz);
    uv = in_uv;
    shade = in_shade;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 uv;
in float shade;
in float distance_to_camera;
uniform vec3 fog_colour;
uniform float fog_near;
uniform float fog_far;
uniform float water_time;
uniform sampler2D atlas;
out vec4 fragment_colour;
void main() {
    float fog = smoothstep(fog_near, fog_far, distance_to_camera);
    vec4 colour = texture(atlas, uv);
    vec3 animated = colour.rgb * shade;
    if (colour.a < 0.9) {
        animated += 0.025 * sin(water_time + distance_to_camera * 0.35);
    }
    fragment_colour = vec4(mix(animated, fog_colour, fog), colour.a);
}
"""

OUTLINE_VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec4 in_colour;
uniform mat4 projection;
uniform mat4 view;
out vec4 colour;
void main() {
    gl_Position = projection * view * vec4(in_position, 1.0);
    colour = in_colour;
}
"""

OUTLINE_FRAGMENT_SHADER = """
#version 330
in vec4 colour;
out vec4 fragment_colour;
void main() {
    fragment_colour = colour;
}
"""

OVERLAY_VERTEX_SHADER = """
#version 330
in vec2 in_position;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

OVERLAY_FRAGMENT_SHADER = """
#version 330
out vec4 fragment_colour;
void main() {
    fragment_colour = vec4(0.96, 0.97, 0.93, 0.92);
}
"""

SKY_VERTEX_SHADER = """
#version 330
in vec2 in_position;
out vec2 uv;
void main() {
    uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 1.0, 1.0);
}
"""

SKY_FRAGMENT_SHADER = """
#version 330
in vec2 uv;
out vec4 fragment_colour;
void main() {
    vec3 horizon = vec3(0.56, 0.74, 0.86);
    vec3 zenith = vec3(0.18, 0.45, 0.76);
    vec3 sky = mix(horizon, zenith, smoothstep(0.0, 0.92, uv.y));
    float sun = 1.0 - smoothstep(0.035, 0.055, distance(uv, vec2(0.78, 0.78)));
    float cloud_band = smoothstep(0.0, 0.08,
        sin(uv.x * 31.0) * 0.025 + sin(uv.x * 53.0) * 0.012 + uv.y - 0.63);
    float clouds = cloud_band * (1.0 - smoothstep(0.64, 0.78, uv.y)) * 0.18;
    sky = mix(sky, vec3(1.0, 0.91, 0.68), sun * 0.9);
    sky = mix(sky, vec3(0.92, 0.95, 0.96), clouds);
    fragment_colour = vec4(sky, 1.0);
}
"""

HUD_VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    uv = in_uv;
}
"""

HUD_FRAGMENT_SHADER = """
#version 330
in vec2 uv;
uniform sampler2D hud_texture;
out vec4 fragment_colour;
void main() {
    fragment_colour = texture(hud_texture, uv);
}
"""
