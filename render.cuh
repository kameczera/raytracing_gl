#ifndef RENDER_CUH
#define RENDER_CUH

#include "color.cuh"
#include "vec3.cuh"
#include "ray.cuh"

using namespace std;

extern GLuint pbo;
extern GLuint tex;
extern struct cudaGraphicsResource* cuda_pbo_resource;

__device__ bool hit_sphere(const point3& center, double radius, const ray& r) {
    
}

__device__ color ray_color(const ray& r) {
    bool t = hit_sphere(point3(0,0,-1), 0.5, r);
    if(t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}
__global__ void renderGradient(uchar4* pixels, int width, int height, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center, float time) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;
    // vec3 new_camera_center = camera_center - vec3(time, time, time);
    vec3 pixel_center = pixel00_loc + (col * pixel_delta_u) + (row * pixel_delta_v);
    vec3 ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);
    color pixel_color = ray_color(r);

    pixels[idx] = write_color(pixel_color);
}

void runCuda(int width, int height, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center, float time) {
    uchar4* dptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_pbo_resource);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    renderGradient<<<grid, block>>>(dptr, width, height, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, time);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    static float time = 0.0f;
    time += .01f;
    double aspect_ratio = 16.0 / 9.0;
    int width = 400;
    int height = int(width / aspect_ratio);
    height = (height < 1) ? 1 : height;

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(width)/height);
    auto camera_center = point3(0, 0, 0);

    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    vec3 pixel_delta_u = viewport_u / width;
    vec3 pixel_delta_v = viewport_v / height;

    vec3 viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    runCuda(width, height, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, time);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
    glutPostRedisplay();
}



#endif