#include <cufft.h>
#include <heffte.h>

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

using Complex = typename heffte::fft_output<double>::type;

void test_cufft (double* pr, Complex* pc, int nx, int ny, int nz,
                 int nxc, int nyc, int nzc)
{
    cudaDeviceSynchronize();
    double t0 = MPI_Wtime();

    cufftHandle plan_fwd, plan_bwd;
    cufftCreate(&plan_fwd);
    cufftCreate(&plan_bwd);
    std::size_t work_size;
    auto status_fwd_plan = cufftMakePlan3d(plan_fwd, nz, ny, nx, CUFFT_D2Z,
                                           &work_size);
    auto status_bwd_plan = cufftMakePlan3d(plan_bwd, nz, ny, nx, CUFFT_Z2D,
                                           &work_size);

    cudaDeviceSynchronize();
    double t1 = MPI_Wtime();

    auto status_fwd_exec = cufftExecD2Z(plan_fwd, pr, (cufftDoubleComplex*)pc);

    cudaDeviceSynchronize();
    double t2 = MPI_Wtime();

    auto status_bwd_exec = cufftExecZ2D(plan_bwd, (cufftDoubleComplex*)pc, pr);

    cudaDeviceSynchronize();
    double t3 = MPI_Wtime();

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_bwd);

    if (status_fwd_plan != CUFFT_SUCCESS ||
        status_bwd_plan != CUFFT_SUCCESS ||
        status_fwd_exec != CUFFT_SUCCESS ||
        status_bwd_exec != CUFFT_SUCCESS)
    {
        std::cout << "Something went wrong in test_cufft\n";
    }

    std::cout << "\ncufft make plan: " << t1-t0 << "\n"
              << "      forward exec: " << t2-t1 << "\n"
              << "      backward exec: " << t3-t2 << "\n"
              << "      total time: " << t3-t0 << std::endl;
}

void test_heffte (double* pr, Complex* pc, int nx, int ny, int nz,
                  int nxc, int nyc, int nzc)
{
    cudaDeviceSynchronize();
    double t0 = MPI_Wtime();

    heffte::fft3d_r2c<heffte::backend::cufft> fft
        ({{0, 0, 0}, {nx-1, ny-1, nz-1}},
         {{0, 0, 0}, {nxc-1, nyc-1, nzc-1}},
         0, MPI_COMM_WORLD);

    cudaDeviceSynchronize();
    double t1 = MPI_Wtime();

    fft.forward(pr, pc);

    cudaDeviceSynchronize();
    double t2 = MPI_Wtime();

    fft.backward(pc, pr);
    cudaDeviceSynchronize();
    double t3 = MPI_Wtime();

    std::cout << "\nheffte make plan: " << t1-t0 << "\n"
              << "       forward exec: " << t2-t1 << "\n"
              << "       backward exec: " << t3-t2 << "\n"
              << "       total time: " << t3-t0 << std::endl;
}

int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    {
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        if (nprocs != 1) {
            std::cout << "This test is for single MPI process only" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    constexpr int nx = 256;
    constexpr int ny = 256;
    constexpr int nz = 256;

    constexpr int nxc = nx/2 + 1;
    constexpr int nyc = ny;
    constexpr int nzc = nz;

    std::vector<double> hv(nx*ny*nz);
    std::generate(hv.begin(), hv.end(), [n=0] () mutable { return n++; });

    double* pr;
    Complex* pc;
    cudaMalloc(&pr, sizeof(double)*nx*ny*nz);
    cudaMalloc(&pc, sizeof(Complex)*nxc*nyc*nzc);
    cudaMemset(pc, 0, sizeof(Complex)*nxc*nyc*nzc);
    cudaMemcpy(pr, hv.data(), sizeof(double)*nx*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    test_cufft(pr, pc, nx, ny, nz, nxc, nyc, nzc);
    test_heffte(pr, pc, nx, ny, nz, nxc, nyc, nzc);

    cudaFree(pr);
    cudaFree(pc);

    MPI_Finalize();
}
