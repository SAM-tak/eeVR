import bpy
import os
import time
import gpu
import numpy as np
from math import sin, cos, tan, ceil, degrees, pi
from datetime import datetime
from gpu_extras.batch import batch_for_shader
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Properties

# Define parts of fragment shader 
commdef = '''
#define PI          3.1415926535897932384626
#define FOVFRAC     %f
#define SIDEFRAC    %f
#define TBFRAC      %f
#define HCLIP       %f
#define VCLIP       %f
#define HMARGIN     %f
#define VMARGIN     %f
#define SMARGIN     %f
#define EXTRUSION   %f
#define INTRUSION   %f

const float SIDEHTEXSCALE = 1 / (SIDEFRAC - INTRUSION);
const float SIDEVTEXSCALE = 1 / (1 + 2 * SMARGIN);
const float TBVTEXSCALE   = 1 / (TBFRAC - INTRUSION);
const float HTEXSCALE     = 1 / (1 + 2 * (HMARGIN + EXTRUSION));
const float VTEXSCALE     = 1 / (1 + 2 * (VMARGIN + EXTRUSION));
const float ACTUALHMARGIN = HMARGIN * HTEXSCALE;
const float ACTUALVMARGIN = VMARGIN * VTEXSCALE;

vec2 tr(vec2 src, vec2 offset, vec2 scale)
{
    return (src + offset) * scale;
}

vec2 tr(vec2 src, vec2 offset, float scale)
{
    return (src + offset) * scale;
}

vec2 tr(vec2 src, float offset, float scale)
{
    return tr(src, vec2(offset, offset), scale);
}

vec2 to_uv(float x, float y)
{
    return tr(vec2(x, y), 1.0, 0.5);
}

vec2 to_uv_right(vec3 pt)
{
    return tr(to_uv(-pt.z/pt.x, pt.y/pt.x), vec2(-INTRUSION, SMARGIN), vec2(SIDEHTEXSCALE, SIDEVTEXSCALE));
}

vec2 to_uv_left(vec3 pt)
{
    return tr(to_uv(-pt.z/pt.x, -pt.y/pt.x), vec2(SIDEFRAC - 1, SMARGIN), vec2(SIDEHTEXSCALE, SIDEVTEXSCALE));
}

vec2 to_uv_top(vec3 pt)
{
    return tr(to_uv(pt.x/pt.y, -pt.z/pt.y), vec2(0, -INTRUSION), vec2(1, TBVTEXSCALE));
}

vec2 to_uv_bottom(vec3 pt)
{
    return tr(to_uv(-pt.x/pt.y, -pt.z/pt.y), vec2(0, TBFRAC - 1), vec2(1, TBVTEXSCALE));
}

vec2 apply_margin(vec2 src)
{
    return tr(src, vec2(HMARGIN + EXTRUSION, VMARGIN + EXTRUSION), vec2(HTEXSCALE, VTEXSCALE));
}

vec2 to_uv_front(vec3 pt)
{
    return apply_margin(to_uv(pt.x/pt.z, pt.y/pt.z));
}

vec2 to_uv_back(vec3 pt)
{
    return apply_margin(to_uv(pt.x/pt.z, -pt.y/pt.z));
}

float atan2(float y, float x)
{
    return x == 0.0 ? sign(y) * 0.5 * PI : atan(y, x);
}

void main() {
'''

dome = '''
    vec2 d = vTexCoord.xy;
    float r = length(d);
    if(r > 1.0) discard;
    
    // Calculate the position on unit sphere
    vec2 dunit = normalize(d);
    float phi = FOVFRAC * PI * r;
    vec3 pt;
    %s
    pt.z = cos(phi);
    float azimuth = atan2(pt.x, pt.z);
'''

domemodes = [
    'pt.xy = dunit * phi;',
    'pt.xy = dunit * sin(phi);',
    'pt.xy = 2.0 * dunit * sin(phi * 0.5);',
    'pt.xy = 2.0 * dunit * tan(phi * 0.5);'
]

equi = '''
    // Calculate the pointing angle
    float azimuth = FOVFRAC * PI * vTexCoord.x;
    float elevation = 0.5 * PI * vTexCoord.y;
    
    // Calculate the position on unit sphere
    vec3 pt;
    pt.x = cos(elevation) * sin(azimuth);
    pt.y = sin(elevation);
    pt.z = cos(elevation) * cos(azimuth);
'''

fetch_setup = '''
    // Select the correct pixel
    // left or right
    float lor = step(abs(pt.y), abs(pt.x)) * step(abs(pt.z), abs(pt.x));
    // top or bottom
    float tob = (1 - lor) * step(abs(pt.z), abs(pt.y));
    // front or back
    float fob = (1 - lor) * (1 - tob);
    float right = step(0, pt.x);
    float up = step(0, pt.y);
    float front = step(0, pt.z);
    float over45 = step(0.25 * PI, abs(azimuth));
    float over135 = step(0.75 * PI, abs(azimuth));
    {
        float angle;
        angle = (fob + tob) * (1 - over45) * front * abs(pt.x/pt.z) * 0.25 * PI;
        angle += (lor + tob) * over45 * (1 - over135) * right * (2 - pt.z/pt.x) * 0.25 * PI;
        angle += (lor + tob) * over45 * (1 - over135) * (1 - right) * (pt.z/pt.x + 2) * 0.25 * PI;
        angle += (fob + tob) * over135 * (1 - front) * (4 - abs(pt.x/pt.z)) * 0.25 * PI;
        if(angle > HCLIP*0.5) discard;
    }
    {
        float near_horizon = step(abs(pt.y), abs(pt.z));
        float angle;
        angle = (fob + lor * near_horizon) * abs(pt.y/pt.z) * 0.25 * PI;
        angle += (tob + lor * (1 - near_horizon)) * (2 - abs(pt.z/pt.y)) * 0.25 * PI;
        if(angle > VCLIP*0.5) discard;
    }
    fragColor = vec4(0.0);
'''

fetch_top_bottom = '''
    fragColor += tob * up * texture(cubeTopImage, to_uv_top(pt));
    fragColor += tob * (1 - up) * texture(cubeBottomImage, to_uv_bottom(pt));
'''

fetch_sides = '''
    vec2 right_uv = to_uv_right(pt);
    fragColor += lor * right * texture(cubeRightImage, right_uv);
    vec2 left_uv = to_uv_left(pt);
    fragColor += lor * (1 - right) * texture(cubeLeftImage, left_uv);
'''

blend_seam_sides = '''
    {
        float range = over45 * (1 - over135);
        
        float alpha = range * right * tob * up * smoothstep(1.0, 0.0, clamp((right_uv.y - 1 + ACTUALVMARGIN) / ACTUALVMARGIN, 0.0, 1.0));
        fragColor = mix(fragColor, texture(cubeRightImage, right_uv), alpha);

        alpha = range * right * tob * (1 - up) * smoothstep(0.0, 1.0, clamp(right_uv.y / ACTUALVMARGIN, 0.0, 1.0));
        fragColor = mix(fragColor, texture(cubeRightImage, right_uv), alpha);

        alpha = range * (1 - right) * tob * up * smoothstep(1.0, 0.0, clamp((left_uv.y - 1 + ACTUALVMARGIN) / ACTUALVMARGIN, 0.0, 1.0));
        fragColor = mix(fragColor, texture(cubeLeftImage, left_uv), alpha);

        alpha = range * (1 - right) * tob * (1 - up) * smoothstep(0.0, 1.0, clamp(left_uv.y / ACTUALVMARGIN, 0.0, 1.0));
        fragColor = mix(fragColor, texture(cubeLeftImage, left_uv), alpha);
    }
'''

fetch_back = '''
    {
        vec2 uv = to_uv_back(pt);
        fragColor += fob * (1 - front) * texture(cubeBackImage, uv);
%s
    }
'''

fetch_front = '''
    {
        vec2 uv = to_uv_front(pt);
        fragColor += fob * front * texture(cubeFrontImage, uv);
%s
    }
'''

blend_seam_front_h = '''
        {
            float in_range = step(0, uv.x) * (1 - step(1, uv.x));
            float alpha = in_range * front * lor * right * smoothstep(1.0, 0.0, clamp((uv.x - 1 + ACTUALHMARGIN) / ACTUALHMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeFrontImage, uv), alpha);
            
            alpha = in_range * front * lor * (1 - right) * smoothstep(0.0, 1.0, clamp(uv.x / ACTUALHMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeFrontImage, uv), alpha);
        }
'''

blend_seam_front_v = '''
        {
            float in_range = step(0, uv.y) * (1 - step(1, uv.y));
            float alpha = in_range * front * tob * up * smoothstep(1.0, 0.0, clamp((uv.y - 1 + ACTUALVMARGIN) / ACTUALVMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeFrontImage, uv), alpha);

            alpha = in_range * front * tob * (1 - up) * smoothstep(0.0, 1.0, clamp(uv.y / ACTUALVMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeFrontImage, uv), alpha);
        }
'''

blend_seam_back_h = '''
        {
            float alpha = (1 - front) * lor * right * smoothstep(1.0, 0.0, clamp((1.0 - uv.x - 1 + ACTUALHMARGIN) / ACTUALHMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeBackImage, uv), alpha);
            
            alpha = (1 - front) * lor * (1 - right) * smoothstep(0.0, 1.0, clamp((1.0 - uv.x) / ACTUALHMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeBackImage, uv), alpha);
        }
'''

blend_seam_back_v = '''
        {
            float alpha = (1 - front) * tob * up * smoothstep(1.0, 0.0, clamp((uv.y - 1 + ACTUALVMARGIN) / ACTUALVMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeBackImage, uv), alpha);

            alpha = (1 - front) * tob * (1 - up) * smoothstep(0.0, 1.0, clamp(uv.y / ACTUALVMARGIN, 0.0, 1.0));
            fragColor = mix(fragColor, texture(cubeBackImage, uv), alpha);
        }
'''

# Define the vertex shader
vertex_shader = '''
void main() {
    vTexCoord = aVertexTextureCoord;
    gl_Position = vec4(aVertexPosition, 1);
}
'''

class Renderer:
    
    def __init__(self, context, is_animation = False, folder = ''):
        
        # Check if the file is saved or not, can cause errors when not saved
        if not bpy.data.is_saved:
            raise PermissionError("Save file before rendering")
        
        props: Properties = context.scene.eeVR

        # Set internal variables for the class
        self.scene = context.scene
        # Get the file extension
        self.fext = os.path.splitext(bpy.context.scene.render.frame_path(preview=True))[-1]
        self.fformat = bpy.context.scene.render.image_settings.file_format.format()
        # save original active object
        self.viewlayer_active_object_origin = context.view_layer.objects.active
        # save original active camera handle
        self.camera_origin = context.scene.camera
        # create a new camera for rendering
        bpy.ops.object.camera_add()
        self.camera = context.object
        self.camera.name = 'eeVR_camera'
        # set new cam active
        context.scene.camera = self.camera
        # set coordinates same as origin by using world matrix already transformed but not location or rotation
        # and always using it to update correct coordinates before rendering
        # no constraints, parent, drivers and keyframes for new cam, now we can handle cameras with those stuff
        self.camera.matrix_world = self.camera_origin.matrix_world
        # transfer key attributes that may affect rendering, conv dis not needed 'cause it is parallel
        self.camera.data.stereo.interocular_distance = self.camera_origin.data.stereo.interocular_distance
        # transfer clip_start & clip_end parameter to new camera
        self.camera.data.clip_start = self.camera_origin.data.clip_start
        self.camera.data.clip_end = self.camera_origin.data.clip_end
        self.path = bpy.path.abspath("//")
        self.is_stereo = context.scene.render.use_multiview
        self.is_animation = is_animation
        is_dome = props.renderModeEnum == 'DOME'
        h_fov = props.get_hfov()
        v_fov = props.get_vfov()
        front_fov = props.get_front_fov()
        ext_front_view = front_fov > pi/2
        render_fov = pi if props.fovModeEnum == '180' else 2 * pi if props.fovModeEnum == '360' else max(h_fov, v_fov)
        self.no_back_image = h_fov <= 3*pi/2
        self.no_side_images = h_fov <= front_fov
        self.no_top_bottom_images = v_fov <= front_fov
        self.seamless = not (context.scene.render.use_multiview and h_fov > pi and props.appliesParallaxForSideAndBack)
        self.remain_temporaries = context.preferences.addons[__package__].preferences.remain_temporaries

        # Calcurate dimension
        self.resolution_x_origin = self.scene.render.resolution_x
        self.resolution_y_origin = self.scene.render.resolution_y
        self.pixel_aspect_x_origin = self.scene.render.pixel_aspect_x
        self.pixel_aspect_y_origin = self.scene.render.pixel_aspect_y
        self.resolution_percentage_origin = self.scene.render.resolution_percentage

        scale = self.resolution_percentage_origin / 100.0
        self.image_size = int(ceil(self.scene.render.resolution_x * scale)), int(ceil(self.scene.render.resolution_y * scale))
        base_resolution = (
            int(ceil(self.image_size[0] * (pi/2 / render_fov))),
            int(ceil(self.image_size[1] * (pi/2 / min(2*pi if is_dome else pi, render_fov))))
        )
        
        # Generate fragment shader code
        fovfrac = render_fov / (2*pi)
        sidefrac = max(0, min(1, (h_fov - pi/2) / pi))
        tbfrac = max(sidefrac, max(0, min(1, (v_fov - pi/2) / pi)))
        
        base_angle = min(h_fov, front_fov)
        
        stitch_margin = 0.0 if self.no_side_images and self.no_top_bottom_images else props.stitchMargin
        margin = max(0.0, 0.5 * (tan(base_angle/2 + stitch_margin) - tan(base_angle/2)))
        extrusion = max(0.0, 0.5 * tan(base_angle/2) - 0.5) if ext_front_view else 0.0
        intrusion = max(0.0, 0.5 - 0.5 * tan(pi/2-base_angle/2)) if ext_front_view else 0.0
        if tbfrac - intrusion <= 0.0 or base_resolution[1]*(tbfrac-intrusion) < 1.0:
            self.no_top_bottom_images = True
        hmargin = 0.0 if self.no_side_images else margin
        vmargin = 0.0 if self.no_top_bottom_images else margin
        smargin = 0.0 if self.no_side_images or not vmargin > 0.0 else max(0.0, 0.5 * (tan(pi/4 + stitch_margin) - tan(pi/4)))
        # print(f"stichAngle {stitch_margin} margin:{margin} hmargin:{hmargin} vmargin:{vmargin} smargin:{smargin} extrusion:{extrusion} intrusion:{intrusion}")
        # print(f"HTEXSCALE:{1 / (1 + 2 * (extrusion + hmargin))} VTEXSCALE:{1 / (1 + 2 * (extrusion + vmargin))}")
        frag_shader = \
           (commdef % (fovfrac, sidefrac, tbfrac, h_fov, v_fov, hmargin, vmargin, smargin, extrusion, intrusion))\
         + (dome % domemodes[int(props.domeMethodEnum)] if is_dome else equi)\
         + fetch_setup\
         + ('' if self.no_top_bottom_images else fetch_top_bottom)\
         + ('' if self.no_side_images else fetch_sides + (blend_seam_sides if vmargin > 0.0 else ''))\
         + ('' if self.no_back_image else (fetch_back % ((blend_seam_back_h if hmargin > 0.0 else '') + (blend_seam_back_v if vmargin > 0.0 else ''))))\
         + (fetch_front % ((blend_seam_front_h if hmargin > 0.0 or ext_front_view else '') + (blend_seam_front_v if vmargin > 0.0 or ext_front_view else '')))\
         + '}'

        shader_info = gpu.types.GPUShaderCreateInfo()
        vert_out = gpu.types.GPUStageInterfaceInfo("eevr")
        vert_out.smooth("VEC2", "vTexCoord")
        shader_info.vertex_in(0, 'VEC3', "aVertexPosition")
        shader_info.vertex_in(1, 'VEC2', "aVertexTextureCoord")
        shader_info.vertex_out(vert_out)
        shader_info.sampler(0, 'FLOAT_2D', "cubeLeftImage")
        shader_info.sampler(1, 'FLOAT_2D', "cubeRightImage")
        shader_info.sampler(2, 'FLOAT_2D', "cubeBottomImage")
        shader_info.sampler(3, 'FLOAT_2D', "cubeTopImage")
        shader_info.sampler(4, 'FLOAT_2D', "cubeBackImage")
        shader_info.sampler(5, 'FLOAT_2D', "cubeFrontImage")
        shader_info.fragment_out(0, 'VEC4', "fragColor")
        shader_info.vertex_source(vertex_shader)
        shader_info.fragment_source(frag_shader)
        self.shader = gpu.shader.create_from_info(shader_info)

        # Set the image name to the current time
        self.start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # get folder name from outside
        self.folder_name = folder
        
        # Get initial camera and output information
        # now origin camera data not need store, and no more need to use empty as proxy
        self.camera_rotation = list(self.camera.rotation_euler)
        self.IPD = self.camera.data.stereo.interocular_distance
        
        # Set camera variables for proper result
        self.camera.data.type = 'PANO'
        self.camera.data.stereo.convergence_mode = 'PARALLEL'
        self.camera.data.stereo.pivot = 'CENTER'
        # transfer depth of field settings
        self.camera.data.dof.use_dof = self.camera_origin.data.dof.use_dof
        if self.camera.data.dof.use_dof:
            self.camera.data.dof.focus_distance = self.camera_origin.data.dof.focus_distance
            self.camera.data.dof.aperture_fstop = self.camera_origin.data.dof.aperture_fstop
            self.camera.data.dof.aperture_blades = self.camera_origin.data.dof.aperture_blades
            self.camera.data.dof.aperture_ratio = self.camera_origin.data.dof.aperture_ratio
            self.camera.data.dof.aperture_rotation = self.camera_origin.data.dof.aperture_rotation

        # setup render targets information
        aspect_ratio = base_resolution[0] / base_resolution[1]

        def make_resolution(src, hscale, vscale, hmargin, vmargin, scale):
            return int(ceil((src[0] * hscale + 2 * hmargin * src[0]) * scale)),\
                int(ceil((src[1] * vscale + 2 * vmargin * src[1]) * scale))

        f_resolution = make_resolution(base_resolution, 1, 1, extrusion+hmargin, extrusion+vmargin, props.frontViewResolution / 100.0)
        b_resolution = make_resolution(base_resolution, 1, 1, extrusion+hmargin, extrusion+vmargin, props.rearViewResolution / 100.0)
        side_resolution = make_resolution(base_resolution, sidefrac-intrusion, 1, 0, smargin, props.sideViewResolution / 100.0)
        top_resolution = make_resolution(base_resolution, 1, tbfrac-intrusion, 0, 0, props.topViewResolution / 100.0)
        bot_resolution = make_resolution(base_resolution, 1, tbfrac-intrusion, 0, 0, props.bottomViewResolution / 100.0)

        fb_angle = base_angle + 2 * stitch_margin
        side_angle = pi/2 + ((2 * stitch_margin) if smargin > 0.0 else 0.0)
        side_shift_scale = 1 / (1 + 2 * smargin)

        self.camera_settings = {
            'top': (0.0, 0.5*(tbfrac-1+intrusion), pi/2, top_resolution[0], top_resolution[1], aspect_ratio),
            'bottom': (0.0, 0.5*(1-tbfrac-intrusion), pi/2, bot_resolution[0], bot_resolution[1], aspect_ratio),
            'right': (0.5*(sidefrac-1+intrusion)*side_shift_scale, 0.0, side_angle, side_resolution[0], side_resolution[1], aspect_ratio),
            'left': (0.5*(1-sidefrac-intrusion)*side_shift_scale, 0.0, side_angle, side_resolution[0], side_resolution[1], aspect_ratio),
            'front': (0.0, 0.0, fb_angle, f_resolution[0], f_resolution[1], aspect_ratio),
            'back': (0.0, 0.0, fb_angle, b_resolution[0], b_resolution[1], aspect_ratio)
        }

        if self.is_stereo:
            self.view_format = self.scene.render.image_settings.views_format
            self.scene.render.image_settings.views_format = 'STEREO_3D'
            self.stereo_mode = self.scene.render.image_settings.stereo_3d_format.display_mode
            if self.stereo_mode in ('SIDEBYSIDE', 'TOPBOTTOM'):
                self.sidebyside = self.stereo_mode == 'SIDEBYSIDE'
            else:
                self.sidebyside = aspect_ratio < 1.5
            props.trueTopBottom = self.stereo_mode == 'TOPBOTTOM'
            self.use_sidebyside_crosseyed = self.scene.render.image_settings.stereo_3d_format.use_sidebyside_crosseyed
            self.scene.render.image_settings.stereo_3d_format.display_mode = 'TOPBOTTOM'
        
        self.direction_offsets = self.find_direction_offsets()
    
    
    def cubemap_to_panorama(self, imageList, outputName):
        
        # Generate the OpenGL shader
        pos = [(-1.0, -1.0, -1.0),  # left,  bottom, back
               (-1.0,  1.0, -1.0),  # left,  top,    back
               (1.0, -1.0, -1.0),   # right, bottom, back
               (1.0,  1.0, -1.0)]   # right, top,    back
        coords = [(-1.0, -1.0),  # left,  bottom
                  (-1.0,  1.0),  # left,  top
                  (1.0, -1.0),   # right, bottom
                  (1.0,  1.0)]   # right, top
        vertexIndices = [(0, 3, 1),(3, 0, 2)]
        
        batch = batch_for_shader(self.shader, 'TRIS', {
            "aVertexPosition": pos,
            "aVertexTextureCoord": coords
        }, indices=vertexIndices)
        
        # Change the color space of all of the images to Linear
        # and load them into OpenGL textures
        textures = []
        for image in imageList:
            image.colorspace_settings.name='Linear'
            tex = gpu.texture.from_image(image)
            textures.append(tex)
        
        # set the size of the final image
        width = self.image_size[0]
        height = self.image_size[1]

        # Create an offscreen render buffer and texture
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            self.shader.bind()

            self.shader.uniform_sampler("cubeFrontImage", textures[0])
            if self.no_side_images:
                if not self.no_top_bottom_images:
                    self.shader.uniform_sampler("cubeBottomImage", textures[1])
                    self.shader.uniform_sampler("cubeTopImage", textures[2])
                    if not self.no_back_image:
                        self.shader.uniform_sampler("cubeBackImage", textures[3])
                else:
                    if not self.no_back_image:
                        self.shader.uniform_sampler("cubeBackImage", textures[1]) # for development purpose
            else:
                self.shader.uniform_sampler("cubeLeftImage", textures[1])
                self.shader.uniform_sampler("cubeRightImage", textures[2])
                if not self.no_top_bottom_images:
                    self.shader.uniform_sampler("cubeBottomImage", textures[3])
                    self.shader.uniform_sampler("cubeTopImage", textures[4])
                    if not self.no_back_image:
                        self.shader.uniform_sampler("cubeBackImage", textures[5])
                else:
                    if not self.no_back_image:
                        self.shader.uniform_sampler("cubeBackImage", textures[3])
            
            # Render the image
            batch.draw(self.shader)

            # Read the resulting pixels into a buffer
            buffer = fb.read_color(0, 0, width, height, 4, 0, 'FLOAT')
            buffer.dimensions = width * height * 4

        # Unload the offscreen texture
        offscreen.free()
        
        # Remove the cubemap textures:
        del textures
        if not self.remain_temporaries:
            for image in imageList:
                bpy.data.images.remove(image)
        
        # Copy the pixels from the buffer to an image object
        if not outputName in bpy.data.images.keys():
            bpy.data.images.new(outputName, width, height)
        imageRes = bpy.data.images[outputName]
        imageRes.scale(width, height)
        imageRes.pixels.foreach_set(buffer)
        return imageRes

    
    def find_direction_offsets(self):
        
        # update location and rotation of our camera from origin one
        self.camera.matrix_world = self.camera_origin.matrix_world
        # Calculate the pointing directions of the camera for each face of the cube
        # Using euler.rotate_axis() to handle, notice that rotation should be done on copies
        eul = self.camera.rotation_euler.copy()
        direction_offsets = {}
        #front
        direction_offsets['front'] = list(eul)
        #back
        eul.rotate_axis('Y', pi)
        direction_offsets['back'] = list(eul)
        #top
        eul = self.camera.rotation_euler.copy()
        eul.rotate_axis('X', pi/2)
        direction_offsets['top'] = list(eul)
        #bottom
        eul.rotate_axis('X', pi)
        direction_offsets['bottom'] = list(eul)
        #left
        eul = self.camera.rotation_euler.copy()
        eul.rotate_axis('Y', pi/2)
        direction_offsets['left'] = list(eul)
        #right
        eul.rotate_axis('Y', pi)
        direction_offsets['right'] = list(eul)
        return direction_offsets
    
    
    def set_camera_direction(self, direction):
        
        # Set the camera to the required postion    
        self.camera.rotation_euler = self.direction_offsets[direction]
        self.camera.data.shift_x = self.camera_settings[direction][0]
        self.camera.data.shift_y = self.camera_settings[direction][1]
        self.camera.data.angle = self.camera_settings[direction][2]
        if self.camera.data.dof.use_dof:
            rate = tan(self.camera_origin.data.angle / 2) / tan(self.camera.data.angle / 2)
            self.camera.data.dof.aperture_fstop = self.camera_origin.data.dof.aperture_fstop * rate
        self.scene.render.resolution_x = self.camera_settings[direction][3]
        self.scene.render.resolution_y = self.camera_settings[direction][4]
        if self.camera_settings[direction][5] >= 1.0:
            self.scene.render.pixel_aspect_x = 1.0
            self.scene.render.pixel_aspect_y = self.camera_settings[direction][5]
        else:
            self.scene.render.pixel_aspect_x = 1 / self.camera_settings[direction][5]
            self.scene.render.pixel_aspect_y = 1.0
        self.scene.render.resolution_percentage = 100
        print(f"{direction} : {self.scene.render.resolution_x} x {self.scene.render.resolution_y} {degrees(self.camera.data.angle):.2f}° [{self.camera.data.shift_x:.3f}, {self.camera.data.shift_y:.3f}] ({self.scene.render.pixel_aspect_x:.2f} : {self.scene.render.pixel_aspect_y:.2f}) fstop={self.camera.data.dof.aperture_fstop:.2f}")


    def clean_up(self, context):

        # Reset all the variables that were changed
        context.view_layer.objects.active = self.viewlayer_active_object_origin
        context.scene.camera = self.camera_origin
        camera = self.camera.data
        bpy.data.objects.remove(self.camera)
        bpy.data.cameras.remove(camera)
        self.scene.render.resolution_x = self.resolution_x_origin
        self.scene.render.resolution_y = self.resolution_y_origin
        self.scene.render.pixel_aspect_x = self.pixel_aspect_x_origin
        self.scene.render.pixel_aspect_y = self.pixel_aspect_y_origin
        self.scene.render.resolution_percentage = self.resolution_percentage_origin
        if self.is_stereo:
            self.scene.render.image_settings.views_format = self.view_format
            self.scene.render.image_settings.stereo_3d_format.display_mode = self.stereo_mode

    def get_render_result(self):
        # Find render result
        render_result : bpy.types.Image = next(image for image in bpy.data.images if image.type == "RENDER_RESULT")

        render_result.gl_load() # これをやっても結局真っ黒

        # Create a GPU texture that shares GPU memory with Blender
        gpu_tex = gpu.texture.from_image(render_result)

        # Read image from GPU
        gpu_tex.read()

        # OR read image into a NumPy array (might be more convenient for later operations)
        fbo = gpu.types.GPUFrameBuffer(color_slots=(gpu_tex,))

        # print('gpu_tex', gpu_tex.width, gpu_tex.height) # 現状(1,1)が帰ってしまって役に立たない

        width = self.scene.render.resolution_x
        height = self.scene.render.resolution_y
        if self.scene.render.use_multiview:
            height *= 2
        buffer_np = np.empty(width * height * 4, dtype=np.float32)
        buffer = gpu.types.Buffer('FLOAT', buffer_np.shape, buffer_np)
        with fbo.bind():
            fbo.read_color(0, 0, width, height, 4, 0, 'FLOAT', data=buffer)

        render_result.gl_free()

        # Now the NumPy array has the pixel data, you can reshape it and/or export it as bytes if you wish
        return buffer_np


    def render_image(self, direction):

        # Render the image and load it into the script
        name = f'temp_img_store_{direction}'

        if self.is_stereo:
            nameL = name + '_L'
            nameR = name + '_R'
            if nameL in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[nameL])
            if nameR in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[nameR])
            
            renderedImageL = bpy.data.images.new(nameL, self.scene.render.resolution_x, self.scene.render.resolution_y)
            renderedImageR = bpy.data.images.new(nameR, self.scene.render.resolution_x, self.scene.render.resolution_y)

            if self.seamless and direction in {'right', 'left'}:
                # If rendering for VR, render the side images separately to avoid seams                # If rendering for VR, render the side images separately to avoid seams

                self.scene.render.use_multiview = False
                tmp_loc = list(self.camera.location)
                camera_angle = self.direction_offsets['front'][2]
                self.camera.location = [tmp_loc[0]+(0.5*self.IPD*cos(camera_angle)),\
                                        tmp_loc[1]+(0.5*self.IPD*sin(camera_angle)),\
                                        tmp_loc[2]]
                bpy.ops.render.render()
                renderedImageL.pixels.foreach_set(self.get_render_result()[:])
                
                self.camera.location = [tmp_loc[0]-(0.5*self.IPD*cos(camera_angle)),\
                                        tmp_loc[1]-(0.5*self.IPD*sin(camera_angle)),\
                                        tmp_loc[2]]
                bpy.ops.render.render()
                renderedImageR.pixels.foreach_set(self.get_render_result()[:])

                self.scene.render.use_multiview = True
                self.camera.location = tmp_loc

            else:
                bpy.ops.render.render()

                # Split the render into two images
                buff = self.get_render_result()
                imageLen = len(buff)
                if self.seamless and direction == 'back':
                    renderedImageL.pixels.foreach_set(buff[imageLen//2:])
                    renderedImageR.pixels.foreach_set(buff[:imageLen//2])
                else:
                    renderedImageR.pixels.foreach_set(buff[imageLen//2:])
                    renderedImageL.pixels.foreach_set(buff[:imageLen//2])
            return renderedImageL, renderedImageR
        else:
            if name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])

            bpy.ops.render.render()
            renderedImage = bpy.data.images.new(name, self.scene.render.resolution_x, self.scene.render.resolution_y)
            renderedImage.pixels.foreach_set(self.get_render_result()[:])
            return renderedImage, None
    
    
    def render_images(self):
        
        # update focus distance if focus object is set
        if self.camera.data.dof.use_dof and self.camera_origin.data.dof.focus_object is not None:
            focus_location = self.camera_origin.data.dof.focus_object.matrix_world.translation
            icm = self.camera_origin.matrix_world.inverted_safe()
            self.camera.data.dof.focus_distance = abs((icm @ focus_location).z)
        
        # Render the images for every direction
        image_list_l = []
        image_list_r = []
        
        directions = ['front']
        if not self.no_side_images:
            directions += ['left', 'right']
        if not self.no_top_bottom_images:
            directions += ['bottom', 'top']
        if not self.no_back_image:
            directions += ['back']

        self.direction_offsets = self.find_direction_offsets()
        for direction in reversed(directions): # I want the results of the front camera to remain in the render window... just that.
            self.set_camera_direction(direction)
            imgl, imgr = self.render_image(direction)
            image_list_l.insert(0, imgl)
            image_list_r.insert(0, imgr)
        
        return image_list_l, image_list_r


    def render_and_save(self):

        frame_step = self.scene.frame_step
        
        # Render the images and return their names
        imageList, imageList2 = self.render_images()
        if self.is_animation:
            image_name = f"frame{self.scene.frame_current:06d}{self.fext}"
        else:
            image_name = f"Render Result {self.start_time}{self.fext}"

        start_time = time.time()
        # Convert the rendered images to equirectangular projection image and save it to the disk
        if self.is_stereo:
            leftImage = self.cubemap_to_panorama(imageList, "Render Left")
            rightImage = self.cubemap_to_panorama(imageList2, "Render Right")

            # If it doesn't already exist, create an image object to store the resulting render
            if not image_name in bpy.data.images.keys():
                imageResult = bpy.data.images.new(image_name, leftImage.size[0], 2 * leftImage.size[1])
            
            imageResult = bpy.data.images[image_name]
            img1arr = np.empty((leftImage.size[1], 4 * leftImage.size[0]), dtype=np.float32)
            leftImage.pixels.foreach_get(img1arr.ravel())
            img2arr = np.empty((rightImage.size[1], 4 * rightImage.size[0]), dtype=np.float32)
            rightImage.pixels.foreach_get(img2arr.ravel())
            if self.sidebyside:
                imageResult.scale(2*leftImage.size[0], leftImage.size[1])
                if self.use_sidebyside_crosseyed:
                    imageResult.pixels.foreach_set(np.concatenate((img1arr, img2arr), axis=1).ravel())
                else:
                    imageResult.pixels.foreach_set(np.concatenate((img2arr, img1arr), axis=1).ravel())
            else:
                imageResult.scale(leftImage.size[0], 2*leftImage.size[1])
                if self.scene.eeVR.isTopRightEye:
                    imageResult.pixels.foreach_set(np.concatenate((img2arr, img1arr)).ravel())
                else:
                    imageResult.pixels.foreach_set(np.concatenate((img1arr, img2arr)).ravel())
            bpy.data.images.remove(leftImage)
            bpy.data.images.remove(rightImage)

        else:
            imageResult = self.cubemap_to_panorama(imageList, "RenderResult")
        
        save_start_time = time.time()
        if self.is_animation:
            # Color Management Settings issue solved by nagadomi
            imageResult.file_format = self.fformat
            imageResult.filepath_raw = self.path+self.folder_name+image_name
            imageResult.save()
            self.scene.frame_set(self.scene.frame_current+frame_step)
        else:
            imageResult.file_format = self.fformat
            imageResult.filepath_raw = self.path+image_name
            imageResult.save()

        print(f'''Saved '{imageResult.filepath_raw}'
 Time : {round(time.time() - start_time, 2)} seconds (Saving : {round(time.time() - save_start_time, 2)} seconds)
 ''')

        bpy.data.images.remove(imageResult)
