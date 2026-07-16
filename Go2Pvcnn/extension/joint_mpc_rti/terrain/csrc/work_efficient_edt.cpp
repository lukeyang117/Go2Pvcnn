#include <torch/extension.h>

#include <vector>

torch::Tensor exact_squared_edt_cuda(torch::Tensor mask);
torch::Tensor semantic_distance_fields_cuda(
    torch::Tensor semantic, int64_t small_id, int64_t large_id, double resolution);
void semantic_distance_fields_out_cuda(
    torch::Tensor semantic,
    torch::Tensor distance,
    torch::Tensor vertical_workspace,
    int64_t small_id,
    int64_t large_id,
    double resolution);
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
    torch::Tensor ready_out);

torch::Tensor exact_squared_edt(torch::Tensor mask) {
  TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must have dtype torch.bool");
  TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
  TORCH_CHECK(mask.dim() == 4, "mask must have shape [B,2,151,151]");
  TORCH_CHECK(mask.size(1) == 2, "mask channel dimension must equal 2");
  TORCH_CHECK(mask.size(2) == 151 && mask.size(3) == 151,
              "mask spatial shape must equal 151x151");
  return exact_squared_edt_cuda(mask);
}

torch::Tensor semantic_distance_fields(
    torch::Tensor semantic, int64_t small_id, int64_t large_id, double resolution) {
  TORCH_CHECK(semantic.is_cuda(), "semantic must be a CUDA tensor");
  TORCH_CHECK(semantic.scalar_type() == torch::kInt64,
              "semantic must have dtype torch.long");
  TORCH_CHECK(semantic.is_contiguous(), "semantic must be contiguous");
  TORCH_CHECK(semantic.dim() == 3, "semantic must have shape [B,151,151]");
  TORCH_CHECK(semantic.size(1) == 151 && semantic.size(2) == 151,
              "semantic spatial shape must equal 151x151");
  TORCH_CHECK(resolution > 0.0, "resolution must be positive");
  return semantic_distance_fields_cuda(semantic, small_id, large_id, resolution);
}

void semantic_distance_fields_out(
    torch::Tensor semantic,
    torch::Tensor distance,
    torch::Tensor vertical_workspace,
    int64_t small_id,
    int64_t large_id,
    double resolution) {
  TORCH_CHECK(semantic.is_cuda() && distance.is_cuda() && vertical_workspace.is_cuda(),
              "semantic, distance, and workspace must be CUDA tensors");
  TORCH_CHECK(semantic.scalar_type() == torch::kInt64,
              "semantic must have dtype torch.long");
  TORCH_CHECK(distance.scalar_type() == torch::kFloat32,
              "distance must have dtype torch.float32");
  TORCH_CHECK(vertical_workspace.scalar_type() == torch::kInt16,
              "vertical workspace must have dtype torch.int16");
  TORCH_CHECK(semantic.is_contiguous() && distance.is_contiguous() &&
                  vertical_workspace.is_contiguous(),
              "semantic, distance, and workspace must be contiguous");
  TORCH_CHECK(semantic.dim() == 3 && semantic.size(1) == 151 && semantic.size(2) == 151,
              "semantic must have shape [B,151,151]");
  TORCH_CHECK(distance.sizes() == vertical_workspace.sizes(),
              "distance and workspace shapes must match");
  TORCH_CHECK(distance.dim() == 4 && distance.size(0) == 2 &&
                  distance.size(1) == semantic.size(0) && distance.size(2) == 151 &&
                  distance.size(3) == 151,
              "distance and workspace must have shape [2,B,151,151]");
  TORCH_CHECK(semantic.device() == distance.device() &&
                  semantic.device() == vertical_workspace.device(),
              "semantic, distance, and workspace must share a device");
  TORCH_CHECK(resolution > 0.0, "resolution must be positive");
  semantic_distance_fields_out_cuda(
      semantic, distance, vertical_workspace, small_id, large_id, resolution);
}

void copy_height_valid(
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
  TORCH_CHECK(height_source.is_cuda() && height_out.is_cuda() && valid_out.is_cuda(),
              "height source/output and valid output must be CUDA tensors");
  TORCH_CHECK(height_source.scalar_type() == torch::kFloat32 &&
                  height_out.scalar_type() == torch::kFloat32,
              "height tensors must have dtype torch.float32");
  TORCH_CHECK(valid_out.scalar_type() == torch::kBool,
              "valid output must have dtype torch.bool");
  TORCH_CHECK(height_source.dim() == 3,
              "height source/output and valid output must have shape [B,N,N]");
  TORCH_CHECK(height_out.is_contiguous() && valid_out.is_contiguous(),
              "height and valid outputs must be contiguous");
  TORCH_CHECK(height_source.sizes() == height_out.sizes() &&
                  height_source.sizes() == valid_out.sizes(),
              "height source/output and valid output shapes must match");
  TORCH_CHECK(height_source.device() == height_out.device() &&
                  height_source.device() == valid_out.device(),
              "height source/output and valid output must share a device");
  const auto batch = height_source.size(0);
  TORCH_CHECK(origin_source.sizes() == torch::IntArrayRef({batch, 3}) &&
                  origin_out.sizes() == origin_source.sizes(),
              "origin tensors must have shape [B,3]");
  TORCH_CHECK(yaw_source.numel() == batch && yaw_out.numel() == batch &&
                  timestamp_source.numel() == batch && timestamp_out.numel() == batch &&
                  version_out.numel() == batch && ready_out.numel() == batch,
              "metadata tensors must have batch B");
  TORCH_CHECK(origin_source.is_cuda() && origin_out.is_cuda() && yaw_source.is_cuda() &&
                  yaw_out.is_cuda() && timestamp_source.is_cuda() &&
                  timestamp_out.is_cuda() && version_out.is_cuda() && ready_out.is_cuda(),
              "metadata tensors must be CUDA tensors");
  TORCH_CHECK(origin_source.scalar_type() == torch::kFloat32 &&
                  origin_out.scalar_type() == torch::kFloat32 &&
                  yaw_source.scalar_type() == torch::kFloat32 &&
                  yaw_out.scalar_type() == torch::kFloat32 &&
                  timestamp_source.scalar_type() == torch::kFloat32 &&
                  timestamp_out.scalar_type() == torch::kFloat32 &&
                  version_out.scalar_type() == torch::kInt64 &&
                  ready_out.scalar_type() == torch::kBool,
              "metadata tensors have invalid dtypes");
  TORCH_CHECK(origin_source.is_contiguous() && origin_out.is_contiguous() &&
                  yaw_source.is_contiguous() && yaw_out.is_contiguous() &&
                  timestamp_source.is_contiguous() && timestamp_out.is_contiguous() &&
                  version_out.is_contiguous() && ready_out.is_contiguous(),
              "metadata tensors must be contiguous");
  TORCH_CHECK(height_source.device() == origin_source.device() &&
                  height_source.device() == origin_out.device() &&
                  height_source.device() == yaw_source.device() &&
                  height_source.device() == yaw_out.device() &&
                  height_source.device() == timestamp_source.device() &&
                  height_source.device() == timestamp_out.device() &&
                  height_source.device() == version_out.device() &&
                  height_source.device() == ready_out.device(),
              "height and metadata tensors must share a device");
  copy_height_valid_cuda(
      height_source, height_out, valid_out, origin_source, origin_out, yaw_source,
      yaw_out, timestamp_source, timestamp_out, version_out, ready_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("exact_squared_edt", &exact_squared_edt,
             "Batched exact squared Euclidean distance transform (CUDA)");
  module.def("semantic_distance_fields", &semantic_distance_fields,
             "Fused semantic exact EDT distances (CUDA)");
  module.def("semantic_distance_fields_out", &semantic_distance_fields_out,
             "Fused semantic exact EDT into fixed output/workspace (CUDA)");
  module.def("copy_height_valid", &copy_height_valid,
             "Copy height and publish finite validity (CUDA)");
}
