# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from glumpy import app, gl, gloo, data, log
from glumpy.transforms import Trackball, PVMProjection, PVMFrustum, Position


vertex = """
uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_normal;
attribute vec3 position;
attribute vec3 normal;
varying vec3 v_normal;
varying vec3 v_position;

void main()
{
    gl_Position = <transform>;
    vec4 P = m_view * m_model* vec4(position, 1.0);
    v_position = P.xyz / P.w;
    v_normal = vec3(m_normal * vec4(normal,0.0));
}
"""

fragment = """
varying vec3 v_normal;
varying vec3 v_position;

const vec3 light_position = vec3(1.0,1.0,1.0);
const vec3 ambient_color = vec3(0.1, 0.0, 0.0);
const vec3 diffuse_color = vec3(0.75, 0.125, 0.125);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);
const float shininess = 128.0;
const float gamma = 2.2;

void main()
{
    vec3 normal= normalize(v_normal);
    vec3 light_direction = normalize(light_position - v_position);
    float lambertian = max(dot(light_direction,normal), 0.0);
    float specular = 0.0;
    if (lambertian > 0.0)
    {
        vec3 view_direction = normalize(-v_position);
        vec3 half_direction = normalize(light_direction + view_direction);
        float specular_angle = max(dot(half_direction, normal), 0.0);
        specular = pow(specular_angle, shininess);
    }
    vec3 color_linear = ambient_color +
                        lambertian * diffuse_color +
                        specular * specular_color;
    vec3 color_gamma = pow(color_linear, vec3(1.0/gamma));
    gl_FragColor = vec4(color_gamma, 1.0);
}
"""

log.info("Loading brain mesh")
vertices,indices = data.get("brain.obj")

brainl = gloo.Program(vertex, fragment)
brainl.bind(vertices)

trackball = Trackball(Position("position"))
brainl['transform'] = trackball
trackball.theta, trackball.phi, trackball.zoom = 80, -135, 15

vertices,indices = data.get("brain.obj")

brainr = gloo.Program(vertex, fragment)
brainr.bind(vertices)

# transform = PVMProjection(Position("position"))
transform = PVMFrustum(Position("position"))
brainr['transform'] = transform

windowl = app.Window(width=1024, height=768)
windowr = app.Window(width=1024, height=768)


def update():

    model = brainl['transform']['model'].reshape(4,4)
    view  = brainl['transform']['view'].reshape(4,4)
    brainl['m_view']  = view
    brainl['m_model'] = model
    brainl['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

    transform.fovy = trackball.zoom
    brainr['transform']['model'] = model
    brainr['transform']['view'] = view
    brainr['m_view']  = view
    brainr['m_model'] = model
    brainr['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)


@windowl.event
def on_draw(dt):
    windowl.clear()
    brainl.draw(gl.GL_TRIANGLES)


@windowr.event
def on_draw(dt):
    windowr.clear()
    brainr.draw(gl.GL_TRIANGLES)


@windowl.event
def on_mouse_drag(x, y, dx, dy, button):
    update()


@windowl.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    update()


@windowr.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    update()


windowl.attach(brainl['transform'])
windowr.attach(brainr['transform'])
app.run()
