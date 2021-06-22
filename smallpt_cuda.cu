//ECE 408

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "structures.cpp"
#include <cuda.h>
#include <curand_kernel.h>
// #include <cuda_runtime.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>
#define double float

#define BLOCK_SIZE 8
#define WIDTH 1024 //width of picture
#define HEIGHT 768 //height of picture
#define NUM_SPHERES 9 //num spheres in picture
#define BILLION 1000000000.0

__constant__ Sphere spheres[NUM_SPHERES];

#define wbCheck(stmt)                                                          \
    do {                                                                       \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "Failed to run stmt %s", #stmt);                   \
            fprintf(stderr, "Got CUDA error ... %s", cudaGetErrorString(err)); \
            return -1;                                                         \
        }                                                                      \
    } while (0)

__host__ __device__ inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }
__host__ inline int toInt(double x){ return int(pow(clamp(x),1/2.2f)*255+.5f); }


__device__ bool intersect(Ray & r, double & t, int & id, Sphere * spheres){
    double d, inf;
    t = 1e20;
    inf = t;    
    for(int i=0; i<NUM_SPHERES; i++){
        d=spheres[i].intersect(r);
        //d = 0;
        if(d!=0&&(d<t)){
            t=d;
            id=i;
        }
    }
    // printf("intersect: %f\n",t);
    if(t<inf)
        return 1;
    else
        return 0;
}
//make this iteritive
__device__ Vec radiance(const Ray &r_, int depth_, Sphere * spheres, curandState_t state){
    double t;
    int id=0;
    Ray r = r_;
    int depth = depth_;
    // printf("in radiance\n");

    Vec cl(0,0,0);
    Vec cf(1,1,1);

    while(1){
        if(!intersect(r, t, id, spheres)) return cl; //if no intersection

        const Sphere &obj = spheres[id];
        Vec x = r.o+r.d*t;
        Vec n = (x-obj.pos).norm();
        Vec nl = n.dot(r.d) < 0 ? n : n*-1;
        Vec f = obj.color;

        // printf("before depth++: %d\n", depth);
        //don't need p for this assignment
        //increase depth
        //depth++;
        cl = cl + cf.mult(obj.emission);

        depth = depth + 1;
        if(depth>5) return cl; //depth is constrined to be less than 5 
        // printf("after depth++: %d\n", depth);

        cf = cf.mult(f);
        if(obj.refl == DIFF){ //diffuse
            // printf("DIFF\n");
            double r1 = 2*M_PI*curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);
            double r2s = sqrt(r2);
            Vec w = nl;
            // printf("1\n");
            Vec u = ((fabs(w.x)>.1?Vec(0,1,0):Vec(1,0,0))%w).norm();
            // printf("2\n");
            Vec v = w%u;
            // printf("3\n");
            Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
            // printf("4\n");
            r = Ray(x,d);
            continue;
        }
        else if(obj.refl==SPEC){ //specular
            // printf("SPEC\n");
            // Vec ret = f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,spheres,state));
            // return obj.emission + ret;
            r = Ray(x,r.d-n*2*n.dot(r.d));
            continue;
        }
        //Refraction
        // printf("REF\n");
        Ray reflRay(x, r.d-n*2*n.dot(r.d));
        bool into = n.dot(nl)>0;  
        double nc = 1;
        double nt = 1.5f;
        double nnt = into ? nc/nt : nt/nc;
        double ddn = r.d.dot(nl);
        double cos2t = 1-nnt*nnt*(1-ddn*ddn);

        if(cos2t<0){ //return obj.emission + f.mult(radiance(reflRay,depth,spheres,state));
            r = reflRay;
            continue;
        }

        Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();

        double a = nt-nc;
        double b = nt+nc;
        double R0 = a*a/(b*b);
        double c = 1-( into ? -ddn : tdir.dot(n));
        double Re = R0+(1-R0)*c*c*c*c*c;
        double Tr = 1-Re;
        double P = .25f+.5f*Re;
        double RP = Re/P;
        double TP = Tr/(1-P);
        
        if(curand_uniform_double(&state) < P){
            cf = cf*RP;
            r = reflRay;
        } else {
            cf = cf*TP;
            r = Ray(x,tdir);
        }
        continue;
        // return obj.emission + f.mult(depth>2 ? (curand_uniform_double(&state) < P ?   // Russian roulette
        // radiance(reflRay,depth,spheres,state)*RP:radiance(Ray(x,tdir),depth,spheres,state)*TP) :
        // radiance(reflRay,depth,spheres,state)*Re+radiance(Ray(x,tdir),depth,spheres,state)*Tr);
    }
}

__global__ void smallptKernel(Ray *cam, Vec *cx, Vec *cy, const int samps, Vec *c, Vec *r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //x value
    int y = blockIdx.y * blockDim.y + threadIdx.y; //y value
    int z = blockIdx.z * blockDim.z + threadIdx.z; //z value for sample number

    //printf("hi %d, %d\n",x,y);
    if(x > WIDTH || y > HEIGHT || z > samps) return; //boundry conditon
    // if(x != 0 || y != 0) return; //only thread 0 can pass
    //make random number
    curandState_t state;
    int id = y*WIDTH+x;
    curand_init(0, id, 0, &state);

    int i = (HEIGHT-y-1)*WIDTH+x;
    // Vec r = Vec(0,0,0);00
    //@@ insert smallpt kernel 
    for (int sy=0; sy<2; sy++){                     // 2x2 subpixel rows
        for (int sx=0; sx<2; sx++, r[i]=Vec(0,0,0)){        // 2x2 subpixel cols
            // for (int s=0; s<samps; s++){
                double r1 = 2*curand_uniform_double(&state);
                double dx = r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
                double r2 = 2*curand_uniform_double(&state);
                double dy = r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
                Vec d = cx[0]*( ( (sx+.5f + dx)/2 + x)/WIDTH - .5f) + cy[0]*( ( (sy+.5f + dy)/2 + y)/HEIGHT - .5f) + cam[0].d;
                d.norm();
                Vec res = radiance(Ray(cam[0].o+d*140,d),0,spheres,state)*(1.0f/samps);
                atomicAdd(&r[i].x,res.x);
                atomicAdd(&r[i].y,res.y);
                atomicAdd(&r[i].z,res.z);
                __syncthreads(); //sync across all z threads
                // r = r + radiance(Ray(cam[0].o+d*140,d),0,spheres,state)*(1.0f/samps);
            // } // Camera rays are pushed ^^^^^ forward to start in interior
        
            if(z==0){
                c[i] = c[i] + Vec(clamp(r[i].x),clamp(r[i].y),clamp(r[i].z))*.25f;
            }
            __syncthreads();
        }
    }
}

int main(int argc, char **argv) {
    struct timespec start, end;
    printf("starting program\n");
    clock_gettime(CLOCK_REALTIME, &start);
    //declare local variables
    int samps = argc == 2 ? atoi(argv[1])/4 : 1; //number of samples
    // int samps = 2; //use this if can't use command line to specifiy sample size
    Ray host_cam[1];
    Vec host_cx[1];
    Vec host_cy[1];
    Vec *host_c; //output data
    Ray *device_cam;
    Vec *device_cx;
    Vec *device_cy;
    Vec *device_c;
    Sphere *device_spheres;
    Vec *device_r;
    //Sphere host_spheres[NUM_SPHERES];

    Sphere host_spheres[] = {//Scene: radius, position, emission, color, material
        Sphere(1e5, Vec( 1e5+1,40.8f,81.6f), Vec(0,0,0),Vec(.75f,.25f,.25f),DIFF),//Left
        Sphere(1e5, Vec(-1e5+99,40.8f,81.6f),Vec(0,0,0),Vec(.25f,.25f,.75f),DIFF),//Rght
        Sphere(1e5, Vec(50,40.8f, 1e5),     Vec(0,0,0),Vec(.75f,.75f,.75f),DIFF),//Back
        Sphere(1e5, Vec(50,40.8f,-1e5+170), Vec(0,0,0),Vec(0,0,0),           DIFF),//Frnt
        Sphere(1e5, Vec(50, 1e5, 81.6f),    Vec(0,0,0),Vec(.75f,.75f,.75f),DIFF),//Botm
        Sphere(1e5, Vec(50,-1e5+81.6f,81.6f),Vec(0,0,0),Vec(.75f,.75f,.75f),DIFF),//Top
        Sphere(16.5f,Vec(27,16.5f,47),       Vec(0,0,0),Vec(1,1,1)*.999f, SPEC),//Mirr
        Sphere(16.5f,Vec(73,16.5f,78),       Vec(0,0,0),Vec(1,1,1)*.999f, REFR),//Glas
        Sphere(600, Vec(50,681.6f-.27f,81.6f),Vec(12,12,12),  Vec(0,0,0), DIFF) //Lite
    }; 

    host_cam[0] = Ray(Vec(50,52,295.6f), Vec(0,-0.042612f,-1).norm());
    host_cx[0] = Vec(WIDTH*.5135f/HEIGHT,0,0);
    host_cy[0] = (host_cx[0]%host_cam[0].d).norm()*.5135f;
    host_c = new Vec[WIDTH*HEIGHT];

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent = (end.tv_sec-start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/BILLION);
	printf("initalizing variables: %f seconds\n",time_spent);
    clock_gettime(CLOCK_REALTIME, &start);

    //Allocate GPU memory
    wbCheck(cudaMalloc((void**)&device_cam,sizeof(Ray)));
    wbCheck(cudaMalloc((void**)&device_cx,sizeof(Vec)));
    wbCheck(cudaMalloc((void**)&device_cy,sizeof(Vec)));
    wbCheck(cudaMalloc((void**)&device_c,sizeof(Vec)*HEIGHT*WIDTH));
    wbCheck(cudaMalloc((void**)&device_r,sizeof(Vec)*HEIGHT*WIDTH));
    wbCheck(cudaMalloc((void**)&device_spheres,sizeof(Sphere)*NUM_SPHERES));

    clock_gettime(CLOCK_REALTIME, &end);
    time_spent = (end.tv_sec-start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/BILLION);
	printf("cudamalloc: %f seconds\n",time_spent);
    clock_gettime(CLOCK_REALTIME, &start);

    //Copying input memory to the GPU
    wbCheck(cudaMemcpy(device_cam,host_cam,sizeof(Ray),cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(device_cx,host_cx,sizeof(Vec),cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(device_cy,host_cy,sizeof(Vec),cudaMemcpyHostToDevice));
    //wbCheck(cudaMemcpy(device_spheres,host_spheres,sizeof(Sphere)*NUM_SPHERES,cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpyToSymbol(spheres,host_spheres,NUM_SPHERES*sizeof(Sphere)));//,0,cudaMemcpyHostToDevice);
    
    clock_gettime(CLOCK_REALTIME, &end);
    time_spent = (end.tv_sec-start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/BILLION);
    printf("memcpy to deivce: %f seconds\n",time_spent);

    //perform CUDA computation
    printf("WIDTH: %d\n",WIDTH);
    printf("HEIGHT: %d\n",HEIGHT);
    printf("BLOCK_SIZE: %d\n",BLOCK_SIZE);

    clock_gettime(CLOCK_REALTIME, &start);
    
//    printf("x for grid: %.02f",ceil(WIDTH*1.0/BLOCK_SIZE));
   dim3 dimGrid(ceil(WIDTH*1.0f/BLOCK_SIZE),ceil(HEIGHT*1.0f/BLOCK_SIZE),ceil(samps*1.0f/BLOCK_SIZE));
   dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);

   smallptKernel<<<dimGrid,dimBlock>>>(device_cam,device_cx,device_cy,samps,device_c,device_r);

    wbCheck(cudaPeekAtLastError());
    wbCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &end);
    time_spent = (end.tv_sec-start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/BILLION);
    printf("ran the kernel %f seconds\n",time_spent);
    clock_gettime(CLOCK_REALTIME, &start);

    //Copying output memory to CPU
    wbCheck(cudaMemcpy(host_c,device_c,sizeof(Vec)*HEIGHT*WIDTH,cudaMemcpyDeviceToHost));
    wbCheck(cudaMemcpy(host_spheres, device_spheres, sizeof(Sphere)*NUM_SPHERES,cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_REALTIME, &end);
    time_spent = (end.tv_sec-start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/BILLION);
	printf("memcpy to host: %f seconds\n",time_spent);
    clock_gettime(CLOCK_REALTIME, &start);

    	//Freeing GPU memory
    wbCheck(cudaFree(device_c));
    wbCheck(cudaFree(device_cam));
    wbCheck(cudaFree(device_cx));
    wbCheck(cudaFree(device_cy));
    wbCheck(cudaFree(device_spheres));

    clock_gettime(CLOCK_REALTIME, &end);
    time_spent = (end.tv_sec-start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/BILLION);
    printf("freeing device allocations: %f seconds\n",time_spent);

    // FILE * fstructs = fopen("structs.txt", "w");

//    for(int i = 0; i<NUM_SPHERES; i++){
//        fprintf(fstructs, "Sphere[%d] rad: %f\n", i, host_spheres[i].rad);
//        fprintf(fstructs, "Sphere[%d] Vec pos: (%f, %f, %f)\n", i, host_spheres[i].pos.x, host_spheres[i].pos.y, host_spheres[i].pos.z);
//        fprintf(fstructs, "Sphere[%d] Vec emission: (%f, %f, %f)\n", i, host_spheres[i].emission.x, host_spheres[i].emission.y, host_spheres[i].emission.z);
//        fprintf(fstructs, "Sphere[%d] Vec color: (%f, %f, %f)\n", i, host_spheres[i].color.x, host_spheres[i].color.y, host_spheres[i].color.z);
//    }

    // Write image to PPM file.
    FILE *f = fopen("image.ppm", "w");         
    fprintf(f, "P3\n%d %d\n%d\n", WIDTH, HEIGHT, 255);
    for (int i=0; i<WIDTH*HEIGHT; i++)
      fprintf(f,"%d %d %d ", toInt(host_c[i].x), toInt(host_c[i].y), toInt(host_c[i].z));

    //Free host memory
    free(host_c);

    return 0;
}
