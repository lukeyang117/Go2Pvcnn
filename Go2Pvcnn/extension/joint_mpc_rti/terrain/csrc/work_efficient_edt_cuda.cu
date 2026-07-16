#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kGridSize = 151;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;
constexpr int kNoSeed = 30000;
constexpr float kEnvelopeInfinity = 1.0e20F;
constexpr float kMaximumSquaredDistance =
    static_cast<float>(2 * (kGridSize - 1) * (kGridSize - 1));

__global__ void vertical_distance_kernel(const bool* __restrict__ mask,
                                         int16_t* __restrict__ vertical,
                                         int image_count) {
  const int image = blockIdx.x;
  const int column = threadIdx.x;
  if (image >= image_count || column >= kGridSize) {
    return;
  }

  const int image_offset = image * kGridSize * kGridSize;
  int nearest = -kNoSeed;
  for (int row = 0; row < kGridSize; ++row) {
    const int index = image_offset + row * kGridSize + column;
    if (mask[index]) {
      nearest = row;
      vertical[index] = 0;
    } else {
      const int delta = row - nearest;
      vertical[index] = nearest <= -kGridSize ? kNoSeed : delta * delta;
    }
  }

  nearest = kNoSeed;
  for (int row = kGridSize - 1; row >= 0; --row) {
    const int index = image_offset + row * kGridSize + column;
    if (mask[index]) {
      nearest = row;
    } else if (nearest < kNoSeed) {
      const int delta = nearest - row;
      vertical[index] = min(vertical[index], delta * delta);
    }
  }
}

__global__ void semantic_vertical_distance_dual_kernel(
    const int64_t* __restrict__ semantic,
    int16_t* __restrict__ vertical,
    int64_t small_id,
    int64_t large_id,
    int batch) {
  const int batch_index = blockIdx.x;
  const int column = threadIdx.x;
  if (batch_index >= batch || column >= kGridSize) {
    return;
  }

  const int semantic_offset = batch_index * kGridSize * kGridSize;
  const int small_offset = batch_index * kGridSize * kGridSize;
  const int large_offset = (batch + batch_index) * kGridSize * kGridSize;
  int nearest_small = -kNoSeed;
  int nearest_large = -kNoSeed;
  for (int row = 0; row < kGridSize; ++row) {
    const int source_index = semantic_offset + row * kGridSize + column;
    const int small_index = small_offset + row * kGridSize + column;
    const int large_index = large_offset + row * kGridSize + column;
    const int64_t semantic_value = semantic[source_index];
    if (semantic_value == small_id) {
      nearest_small = row;
      vertical[small_index] = 0;
    } else {
      const int delta = row - nearest_small;
      vertical[small_index] = nearest_small <= -kGridSize ? kNoSeed : delta * delta;
    }
    if (semantic_value == large_id) {
      nearest_large = row;
      vertical[large_index] = 0;
    } else {
      const int delta = row - nearest_large;
      vertical[large_index] = nearest_large <= -kGridSize ? kNoSeed : delta * delta;
    }
  }

  nearest_small = kNoSeed;
  nearest_large = kNoSeed;
  for (int row = kGridSize - 1; row >= 0; --row) {
    const int source_index = semantic_offset + row * kGridSize + column;
    const int small_index = small_offset + row * kGridSize + column;
    const int large_index = large_offset + row * kGridSize + column;
    const int64_t semantic_value = semantic[source_index];
    if (semantic_value == small_id) {
      nearest_small = row;
    } else if (nearest_small < kNoSeed) {
      const int delta = nearest_small - row;
      vertical[small_index] = min(vertical[small_index], delta * delta);
    }
    if (semantic_value == large_id) {
      nearest_large = row;
    } else if (nearest_large < kNoSeed) {
      const int delta = nearest_large - row;
      vertical[large_index] = min(vertical[large_index], delta * delta);
    }
  }
}

__global__ void copy_height_valid_kernel(const float* __restrict__ source,
                                         float* __restrict__ output,
                                         bool* __restrict__ valid,
                                         const float* __restrict__ origin_source,
                                         float* __restrict__ origin_out,
                                         const float* __restrict__ yaw_source,
                                         float* __restrict__ yaw_out,
                                         const float* __restrict__ timestamp_source,
                                         float* __restrict__ timestamp_out,
                                         int64_t* __restrict__ version_out,
                                         bool* __restrict__ ready_out,
                                         int64_t element_count,
                                         int64_t batch_size,
                                         int64_t size_1,
                                         int64_t size_2,
                                         int64_t stride_0,
                                         int64_t stride_1,
                                         int64_t stride_2) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count) {
    const int64_t column = index % size_2;
    const int64_t row = (index / size_2) % size_1;
    const int64_t batch = index / (size_1 * size_2);
    const float value = source[batch * stride_0 + row * stride_1 + column * stride_2];
    output[index] = value;
    valid[index] = isfinite(value);
  }
  if (index < batch_size) {
    origin_out[index * 3] = origin_source[index * 3];
    origin_out[index * 3 + 1] = origin_source[index * 3 + 1];
    origin_out[index * 3 + 2] = origin_source[index * 3 + 2];
    yaw_out[index] = yaw_source[index];
    timestamp_out[index] = timestamp_source[index];
    version_out[index] += 1;
    ready_out[index] = true;
  }
}

__global__ void horizontal_envelope_kernel(const int16_t* __restrict__ vertical,
                                           float* __restrict__ output,
                                           int image_count,
                                           float resolution,
                                           bool square_root_output) {
  const int image = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (image >= image_count) {
    return;
  }

  __shared__ int envelope_index[kWarpsPerBlock][kGridSize];
  __shared__ float envelope_break[kWarpsPerBlock][kGridSize + 1];
  __shared__ int envelope_size[kWarpsPerBlock];

  const int image_offset = image * kGridSize * kGridSize;
  for (int row = warp; row < kGridSize; row += kWarpsPerBlock) {
    const int row_offset = image_offset + row * kGridSize;
    if (lane == 0) {
      int size = -1;
      for (int column = 0; column < kGridSize; ++column) {
        const int value = vertical[row_offset + column];
        if (value >= kNoSeed) {
          continue;
        }
        float intersection = -kEnvelopeInfinity;
        while (size >= 0) {
          const int previous_column = envelope_index[warp][size];
          const int previous_value = vertical[row_offset + previous_column];
          intersection =
              static_cast<float>((value + column * column) -
                                 (previous_value + previous_column * previous_column)) /
              static_cast<float>(2 * (column - previous_column));
          if (intersection > envelope_break[warp][size]) {
            break;
          }
          --size;
        }
        ++size;
        envelope_index[warp][size] = column;
        envelope_break[warp][size] = size == 0 ? -kEnvelopeInfinity : intersection;
        envelope_break[warp][size + 1] = kEnvelopeInfinity;
      }
      envelope_size[warp] = size;
    }
    __syncwarp();

    const int size = envelope_size[warp];
    for (int query = lane; query < kGridSize; query += 32) {
      if (size < 0) {
        const float squared = kMaximumSquaredDistance;
        output[row_offset + query] =
            square_root_output ? sqrtf(squared) * resolution : squared;
        continue;
      }
      int lower = 0;
      int upper = size + 1;
      while (lower + 1 < upper) {
        const int middle = (lower + upper) >> 1;
        if (envelope_break[warp][middle] <= static_cast<float>(query)) {
          lower = middle;
        } else {
          upper = middle;
        }
      }
      const int seed_column = envelope_index[warp][lower];
      const int delta = query - seed_column;
      const float squared =
          static_cast<float>(vertical[row_offset + seed_column] + delta * delta);
      output[row_offset + query] =
          square_root_output ? sqrtf(squared) * resolution : squared;
    }
    __syncwarp();
  }
}

}  // namespace

torch::Tensor exact_squared_edt_cuda(torch::Tensor mask) {
  const c10::cuda::CUDAGuard device_guard(mask.device());
  auto output = torch::empty(mask.sizes(), mask.options().dtype(torch::kFloat32));
  auto vertical = torch::empty(mask.sizes(), mask.options().dtype(torch::kInt16));
  const int image_count = static_cast<int>(mask.size(0) * mask.size(1));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  vertical_distance_kernel<<<image_count, 256, 0, stream>>>(
      mask.data_ptr<bool>(), vertical.data_ptr<int16_t>(), image_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  horizontal_envelope_kernel<<<image_count, kThreadsPerBlock, 0, stream>>>(
      vertical.data_ptr<int16_t>(), output.data_ptr<float>(), image_count, 1.0F, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}


torch::Tensor semantic_distance_fields_cuda(
    torch::Tensor semantic, int64_t small_id, int64_t large_id, double resolution) {
  const c10::cuda::CUDAGuard device_guard(semantic.device());
  const int batch = static_cast<int>(semantic.size(0));
  const int image_count = batch * 2;
  auto field_sizes = std::vector<int64_t>{2, batch, kGridSize, kGridSize};
  auto float_options = semantic.options().dtype(torch::kFloat32);
  auto int_options = semantic.options().dtype(torch::kInt16);
  auto vertical = torch::empty(field_sizes, int_options);
  auto distance = torch::empty(field_sizes, float_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  semantic_vertical_distance_dual_kernel<<<batch, 256, 0, stream>>>(
      semantic.data_ptr<int64_t>(), vertical.data_ptr<int16_t>(), small_id, large_id,
      batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  horizontal_envelope_kernel<<<image_count, kThreadsPerBlock, 0, stream>>>(
      vertical.data_ptr<int16_t>(), distance.data_ptr<float>(), image_count,
      static_cast<float>(resolution), true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return distance;
}


void semantic_distance_fields_out_cuda(
    torch::Tensor semantic,
    torch::Tensor distance,
    torch::Tensor vertical_workspace,
    int64_t small_id,
    int64_t large_id,
    double resolution) {
  const c10::cuda::CUDAGuard device_guard(semantic.device());
  const int batch = static_cast<int>(semantic.size(0));
  const int image_count = batch * 2;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  semantic_vertical_distance_dual_kernel<<<batch, 256, 0, stream>>>(
      semantic.data_ptr<int64_t>(), vertical_workspace.data_ptr<int16_t>(), small_id,
      large_id, batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  horizontal_envelope_kernel<<<image_count, kThreadsPerBlock, 0, stream>>>(
      vertical_workspace.data_ptr<int16_t>(), distance.data_ptr<float>(), image_count,
      static_cast<float>(resolution), true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void copy_height_valid_cuda(
    torch::Tensor height_source,
    torch::Tensor height_out,
    torch::Tensor valid_out,
    torch::Tensor origin_source,
    torch::Tensor origin_out,
    torch::Tensor yaw_source,
    torch::Tensor yaw_out,
    torch::Tensor timestamp_source,
    torch::Tensor timestamp_out,
    torch::Tensor version_out,
    torch::Tensor ready_out) {
  const c10::cuda::CUDAGuard device_guard(height_source.device());
  const int64_t element_count = height_source.numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_height_valid_kernel<<<static_cast<int>((element_count + 255) / 256), 256, 0, stream>>>(
      height_source.data_ptr<float>(), height_out.data_ptr<float>(),
      valid_out.data_ptr<bool>(), origin_source.data_ptr<float>(),
      origin_out.data_ptr<float>(), yaw_source.data_ptr<float>(),
      yaw_out.data_ptr<float>(), timestamp_source.data_ptr<float>(),
      timestamp_out.data_ptr<float>(), version_out.data_ptr<int64_t>(),
      ready_out.data_ptr<bool>(), element_count, height_source.size(0), height_source.size(1),
      height_source.size(2), height_source.stride(0), height_source.stride(1),
      height_source.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
