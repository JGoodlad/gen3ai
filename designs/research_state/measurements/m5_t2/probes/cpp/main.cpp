// Load an AOTInductor package WITHOUT Python and run it on raw input files.
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
int main(int argc, char** argv) {
  std::string pkg = argv[1], dir = argv[2];
  auto t0 = std::chrono::steady_clock::now();
  torch::inductor::AOTIModelPackageLoader loader(pkg);
  auto t1 = std::chrono::steady_clock::now();
  std::ifstream man(dir + "/manifest.txt");
  std::vector<torch::Tensor> in;
  std::string line;
  while (std::getline(man, line)) {
    std::istringstream ss(line); std::string f, code; int nd; ss >> f >> code >> nd;
    std::vector<int64_t> shape(nd); for (auto& s : shape) ss >> s;
    auto dt = code == "f32" ? torch::kFloat32 : code == "b8" ? torch::kBool : torch::kInt64;
    auto t = torch::empty(shape, torch::TensorOptions().dtype(dt));
    std::ifstream bin(dir + "/" + f, std::ios::binary);
    bin.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    in.push_back(t.to(torch::kCUDA));
  }
  auto out = loader.run(in);
  torch::cuda::synchronize();
  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < 20; ++i) out = loader.run(in);
  torch::cuda::synchronize();
  auto t3 = std::chrono::steady_clock::now();
  auto logp = out[0].cpu().contiguous(), v = out[1].cpu().contiguous();
  std::ofstream(dir + "/cpp_logp.bin", std::ios::binary).write((char*)logp.data_ptr(), logp.nbytes());
  std::ofstream(dir + "/cpp_v.bin", std::ios::binary).write((char*)v.data_ptr(), v.nbytes());
  std::cout << "load_ms " << std::chrono::duration<double, std::milli>(t1 - t0).count()
            << " run_ms " << std::chrono::duration<double, std::milli>(t3 - t2).count() / 20
            << " outputs " << out.size() << std::endl;
}
