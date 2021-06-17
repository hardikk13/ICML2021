// Compile with:
// nvcc -o strut_lattice_helmet -arch=sm_37 --expt-relaxed-constexpr strut_lattice_helmet.cu

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "stdio.h"

#include "neuralNetwork.hh"
#include "layers/denseLayer.hh"
#include "MarchingCubes.h"

// CSG Operations
__device__
float sdfOpIntersect(float f1, float f2) {
  return std::max(f1, f2);
}

__device__
float sdfOpUnion(float distA, float distB) {
    return min(distA, distB);
}

__device__
float sdfOpDifference(float distA, float distB) {
    return max(distA, -distB);
}

// SDF functions

struct Point {
  float x = 0;
  float y = 0;
  float z = 0;
};

// See https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
__device__
float Capsule(float x, float y, float z, const Point& p0, const Point& p1, float r) {
  const float pa_x = x - p0.x;
  const float pa_y = y - p0.y;
  const float pa_z = z - p0.z;
  const float ba_x = p1.x - p0.x;
  const float ba_y = p1.y - p0.y;
  const float ba_z = p1.z - p0.z;
  const float pa_dot_ba = pa_x*ba_x + pa_y*ba_y + pa_z*ba_z;
  const float ba_sq_norm = ba_x*ba_x + ba_y*ba_y + ba_z*ba_z;
  // std::clamp is C++17. nvcc using -arch=sm_37 doesn't compile...
  // const float h = std::clamp(pa_dot_ba/ba_sq_norm, 0.0, 1.0);
  float h = pa_dot_ba/ba_sq_norm;
  if (h > 1) h = 1;
  if (h < 0) h = 0;
  const float diff_x = pa_x - ba_x*h;
  const float diff_y = pa_y - ba_y*h;
  const float diff_z = pa_z - ba_z*h;
  const float diff_norm = std::sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
  return diff_norm - r;
}

__device__
float SphericalNode(float x, float y, float z, const Point& p0, float r) {
  const float diff_x = x - p0.x;
  const float diff_y = x - p0.y;
  const float diff_z = x - p0.z;
  return std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) - r;
}

__device__
float StrutLattice(float x, float y, float z, float* strut_pts, std::uint32_t num_edges,
                   std::uint32_t* strut_edges, float r) {
  Point p0;
  p0.x = strut_pts[3*strut_edges[0]];
  p0.y = strut_pts[3*strut_edges[0] + 1];
  p0.z = strut_pts[3*strut_edges[0] + 2];
  Point p1;
  p1.x = strut_pts[3*strut_edges[1]];
  p1.y = strut_pts[3*strut_edges[1] + 1];
  p1.z = strut_pts[3*strut_edges[1] + 2];
  float f = Capsule(x, y, z, p0, p1, r);
  for (std::uint32_t i = 1; i < num_edges; ++i) {
    const std::uint32_t idx0 = 2*i;
    const std::uint32_t idx1 = idx0 + 1;
    p0.x = strut_pts[3*strut_edges[idx0]];
    p0.y = strut_pts[3*strut_edges[idx0] + 1];
    p0.z = strut_pts[3*strut_edges[idx0] + 2];
    p1.x = strut_pts[3*strut_edges[idx1]];
    p1.y = strut_pts[3*strut_edges[idx1] + 1];
    p1.z = strut_pts[3*strut_edges[idx1] + 2];
    const float f_other = Capsule(x, y, z, p0, p1, r);
    const float f0_sphere = SphericalNode(x, y, z, p0, r * 10.f);
    const float f1_sphere = SphericalNode(x, y, z, p1, r * 10.f);
    f = sdfOpUnion(f, sdfOpUnion(sdfOpUnion(f_other, f0_sphere), f1_sphere));
  }
  return f;
}


void PrintMatrix(Matrix& batch, Matrix& sdf) {
    batch.copyDeviceToHost();
    sdf.copyDeviceToHost();
    printf("BATCH: \n");
    for (int i = 0; i < sdf.size(); ++i) {
       printf("\t %d: (%f, %f, %f), %f \n", i , batch[3 * i], batch[3 * i + 1], batch[3 * i + 2], sdf[i]);
    }
    // for (int i = 0; i < batch.size()-2; i += 3) {
    //     printf("\t %d: (%f, %f, %f), %f \n", i/3, batch[i], batch[i+1], batch[i+2], sdf[i / 3]);
    // }
    std::cout << "\n\n";
}

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
 #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__global__
void CreateBatch(float min_x, float min_y, float min_z, float max_x,
                 float max_y, float max_z, std::uint64_t num_pts_x,
                 std::uint64_t num_pts_y, std::uint64_t num_pts_z,
                 float* d_points) {
  const std::uint64_t index = threadIdx.x + blockIdx.x*blockDim.x;
  // printf("Index %ld\n", index);
  const std::uint64_t idx = index % num_pts_x;
  const std::uint64_t idy = (index / num_pts_x) % num_pts_y;
  const std::uint64_t idz = index / (num_pts_x * num_pts_y);
  if (idz >= num_pts_z) return;
  const float x = min_x + idx * (max_x - min_x) / (num_pts_x - 1);
  const float y = min_y + idy * (max_y - min_y) / (num_pts_y - 1);
  const float z = min_z + idz * (max_z - min_z) / (num_pts_z - 1);
  // printf("Index %ld, x=%f, y=%f, z=%f\n", index, x, y, z);
  // batch = Matrix(Shape(int(numInputs*COLOR_MASK_VAL), imageSize)); 
  d_points[3 * index] = x;
  d_points[3 * index + 1] = y;
  d_points[3 * index + 2] = z;
}

__global__
void EvaluateGrid(float* strut_pts, std::uint32_t num_edges, std::uint32_t* strut_edges, float r, float min_x,
                  float min_y, float min_z, float max_x, float max_y, float max_z, std::uint64_t num_pts_x,
                  std::uint64_t num_pts_y, std::uint64_t num_pts_z, const float* neural_implicit_sdf, float* results_gpu) {
  // Convert points into batch matrix
  // Evalulate sdf for all of them.
  // then pass the sdf here.
  const std::uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  const std::uint64_t idy = blockIdx.y*blockDim.y + threadIdx.y;
  const std::uint64_t idz = blockIdx.z*blockDim.z + threadIdx.z;
  // Index into results, which is a 1D array
  const std::uint64_t index = idx + idy * num_pts_x + idz * num_pts_x * num_pts_y;
  if (idx >= num_pts_x) return;
  if (idy >= num_pts_y) return;
  if (idz >= num_pts_z) return;
  const float x = min_x + idx * (max_x - min_x) / (num_pts_x - 1);
  const float y = min_y + idy * (max_y - min_y) / (num_pts_y - 1);
  const float z = min_z + idz * (max_z - min_z) / (num_pts_z - 1);
  results_gpu[index] = StrutLattice(x, y, z, strut_pts, num_edges, strut_edges, r);
  // tanh(neural_implicit_sdf[index]);
  // printf("Index %ld, x=%f, y=%f, z=%f, sdf=%f\n", index, x, y, z, results_gpu[index]);
  // StrutLattice(x, y, z, strut_pts, num_edges, strut_edges, r);
  // tanh(neural_implicit_sdf[index]);
      // sdfOpIntersect(tanh(neural_implicit_sdf[index]), StrutLattice(x, y, z, strut_pts, num_edges, strut_edges, r));
}

void GenerateLattice(dim3 gridSize, dim3 blockSize,
                     float* strut_pts, std::uint32_t num_edges,
                     std::uint32_t* strut_edges, float r, float min_x, float min_y, float min_z,
                     float max_x, float max_y, float max_z, std::uint64_t num_pts_x,
                     std::uint64_t num_pts_y, std::uint64_t num_pts_z, NeuralNetwork& nn,
                     float* results_gpu) {
  Matrix batch, sdf;
  // batch = Matrix(Shape(3, num_pts_x * num_pts_y * num_pts_z));
  // batch.allocateMemory();
  std::cout << "# of points to get sdf at: " << num_pts_x * num_pts_y * num_pts_z
            << ", size of batch" << batch.size() << std::endl;
  std::cout << gridSize.x << "," << gridSize.y << "," << gridSize.z << std::endl;
  std::cout << blockSize.x << "," << blockSize.y << "," << blockSize.z << std::endl;
  // CreateBatch<<<gridSize.x, blockSize.x>>> (
  //       min_x, min_y, min_z,
  //       max_x, max_y, max_z,
  //       num_pts_x, num_pts_y, num_pts_z,
  //       batch.deviceData.get());  
  // std::cout << "About to evaulate nn" << std::endl;
  // sdf = nn.forward(batch, int(num_pts_x * num_pts_y * num_pts_z * 3));
  // PrintMatrix(batch, sdf);
  // const float* d_sdf = sdf.deviceData.get();
  EvaluateGrid<<<gridSize, blockSize>>>(
        strut_pts, num_edges, strut_edges, r,
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        num_pts_x, num_pts_y, num_pts_z,
        (const float *) sdf.deviceData.get(), results_gpu);
  // cudaDeviceSynchronize();        
}

// File IO

// Read node and edges file.
// In the future this should just read a VTU file directly.

// Numbers should be space and newline delimited only.
template<typename T>
std::vector<T> ReadFile(const std::string& path) {
  std::ifstream file(path);
  std::vector<T> pts;
  T val = 0;
  while (file >> val) {
    pts.push_back(val);
  }
  file.close();
  return pts;
}

// Helper functions on CPU

void Check(cudaError_t s) {
  if (s != cudaSuccess) {
    std::cout << "cudaError_t: " << s << "\n";
    assert(s == cudaSuccess);
  }
}

void Triangulate(std::uint32_t num_pts_x, std::uint32_t num_pts_y, std::uint32_t num_pts_z,
                 const std::string& out_path, float* grid_pts) {
  MarchingCubes mc(num_pts_x, num_pts_y, num_pts_z);
  mc.init_all() ;
  mc.set_ext_data(grid_pts);
  mc.run();
  mc.clean_temps();
  const bool write_binary = true;
  mc.writePLY(out_path.c_str(), write_binary);
  mc.clean_all();
}

// Args (starting from index 1 are):
// strut_pts_file_path
// strut_edges_file_path
// strut_diameter_mm
// bbox_min_x
// bbox_min_y
// bbox_min_z
// bbox_max_x
// bbox_max_y
// bbox_max_z
// voxel_size
// neuralgeometryinput
// data_out_path
int main(int argc, char** argv) {
  // Read neaural geometry as input
  NeuralNetwork nn;
  std::cout << "Trying to load a neural net from: " << argv[11] << std::endl;
  bool ok = nn.load(argv[11]);
  if (!ok) {
    printf("Failed to initialize model (%s)... exiting \n", argv[11]);
    return 0;
  }     
  if (false) {
    Matrix test;
    test = Matrix(Shape(3, 1), false);
    test.allocateMemory();
    float* test_data = test.hostData.get();
    // -0.560611,-0.063418,0.701209, sdf 0.364737 
    // test_data[0] = -0.560611;
    // test_data[1] = -0.063418;
    // test_data[2] = 0.701209;
    test_data[0] = -0.032438;
    test_data[1] = 0.37253597053168713;
    test_data[2] = 0.45408282803863353;
    test.copyHostToDevice();
    // std::cout << "Here here" << std::endl;
    Matrix ss = nn.forward(test, int(1*3));
    cudaDeviceSynchronize();
    ss.copyDeviceToHost();
    const float* data_ss = ss.hostData.get();
    std::cout << "Shape of the sdf:" << ss.size() << std::endl;
    // std::cout << "sdf at (-0.560611,-0.063418,0.701209) is "
    //           << data_ss[0] << ", " << tanh(data_ss[0]) << std::endl;
    // This should  be -ve answer. Start from here.
    std::cout << "sdf at (-0.032438,0.37253597053168713,0.45408282803863353) is "
              << data_ss[0] << ", " << tanh(data_ss[0]) << std::endl;
    exit(0);
  }
  std::cout << "Loaded the neural net" << std::endl;
  std::cout << std::flush;
  auto start = std::chrono::steady_clock::now();
  // Read in pts and edges
  std::cout << "Reading points and edge files\n";
  assert(argc == 12);
  const std::string pts_path(argv[1]);
  const std::string edges_path(argv[2]);
  const std::vector<float> pts = ReadFile<float>(pts_path);
  std::cout << "Read in the file: " << pts_path << std::endl;
  const std::vector<std::uint32_t> edges = ReadFile<std::uint32_t>(edges_path);
  std::cout << "Read in the file: " << edges_path << " " << edges.size() << std::endl;
  float strut_pts[pts.size()];
  for (std::uint32_t i = 0; i < pts.size(); ++i) {
    strut_pts[i] = pts[i];
  }
  std::cout << "Strut_pts " << pts.size() << std::endl;
  std::uint32_t strut_edges[edges.size()];
  std::cout << "strut_edges " << edges.size() << std::endl;
  for (std::uint32_t i = 0; i < edges.size(); ++i) {
    strut_edges[i] = edges[i];
  }
  std::cout << "Done reading" << std::endl;
  const std::uint32_t num_edges = edges.size() / 2;
  // Copy strut points and edge data to GPU
  float* strut_pts_gpu;
  std::uint32_t* strut_edges_gpu;
  Check(cudaMalloc((void**)&strut_pts_gpu, pts.size()*sizeof(float)));
  Check(cudaMemcpy(strut_pts_gpu, strut_pts, pts.size()*sizeof(float), cudaMemcpyHostToDevice));
  Check(cudaMalloc((void**)&strut_edges_gpu, edges.size()*sizeof(std::uint32_t)));
  Check(cudaMemcpy(strut_edges_gpu, strut_edges, edges.size()*sizeof(std::uint32_t), cudaMemcpyHostToDevice));
  // Get remaining args
  const float strut_diameter_mm = std::stof(argv[3]);
  const float bbox_min_x = std::stof(argv[4]);
  const float bbox_min_y = std::stof(argv[5]);
  const float bbox_min_z = std::stof(argv[6]);
  const float bbox_max_x = std::stof(argv[7]);
  const float bbox_max_y = std::stof(argv[8]);
  const float bbox_max_z = std::stof(argv[9]);
  const float voxel_size = std::stof(argv[10]);
  const std::uint64_t num_pts_x = static_cast<int>(std::ceil((bbox_max_x - bbox_min_x)/voxel_size));
  const std::uint64_t num_pts_y = static_cast<int>(std::ceil((bbox_max_y - bbox_min_y)/voxel_size));
  const std::uint64_t num_pts_z = static_cast<int>(std::ceil((bbox_max_z - bbox_min_z)/voxel_size));
  const std::uint64_t num_pts_total_uint64 = num_pts_x * num_pts_y * num_pts_z;
  assert(num_pts_total_uint64 < UINT64_MAX);
  const std::uint32_t num_pts_total = static_cast<std::uint64_t>(num_pts_total_uint64);
  // Call GPU kernel
  std::cout << "Generating strut lattice on GPU\n";
  std::cout << "Num total points to evaluate: " << num_pts_total << std::endl;
  
  // Max of 1024 threads total per block. Make each block 32x32x1
  dim3 kBlockSize(32, 32, 1);
  const int num_blocks_x = static_cast<int>(std::ceil(static_cast<float>(num_pts_x)/kBlockSize.x));
  const int num_blocks_y = static_cast<int>(std::ceil(static_cast<float>(num_pts_y)/kBlockSize.y));
  const int num_blocks_z = static_cast<int>(std::ceil(static_cast<float>(num_pts_z)/kBlockSize.z));
  dim3 gridSize(num_blocks_x, num_blocks_y, num_blocks_z);
  float* results_gpu;
  Check(cudaMalloc((void**)&results_gpu, num_pts_total*sizeof(float)));
  const float r = strut_diameter_mm / 2.0;
  GenerateLattice(gridSize, kBlockSize, 
                  strut_pts_gpu, num_edges, strut_edges_gpu, r,
                  bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x,
                  bbox_max_y, bbox_max_z, num_pts_x, num_pts_y,
                  num_pts_z, nn, results_gpu);
  // cudaDeviceSynchronize();
  // EvaluateGrid<<<num_blocks, threads_per_block>>>(strut_pts_gpu, num_edges, strut_edges_gpu, r,
  //                                                 bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x,
  //                                                 bbox_max_y, bbox_max_z, num_pts_x, num_pts_y,
  //                                                 num_pts_z, nn, results_gpu);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed_seconds = end - start;
  std::cout << "Time from start to end of GPU EvaluateGrid: " << elapsed_seconds.count() << "s\n";
  // Copy GPU data back to CPU
  std::cout << "Copying results back to CPU\n";
  float* results = new float[num_pts_total];
  Check(cudaMemcpy(results, results_gpu, num_pts_total*sizeof(float), cudaMemcpyDeviceToHost));
  Check(cudaFree(results_gpu));
  Check(cudaFree(strut_pts_gpu));
  Check(cudaFree(strut_edges_gpu));
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "Time from start to end of GPU portion: " << elapsed_seconds.count() << "s\n";
  std::cout << "Finished with CUDA\n";
  std::cout << std::endl;
  Triangulate(num_pts_x, num_pts_y, num_pts_z, argv[12], results);
  // delete[] results;
  return 0;
}
