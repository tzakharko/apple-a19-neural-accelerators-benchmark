import Metal

func tensor_matmul_cpu<T: TensorData, Accumulator: TensorData>(left: Tensor<T>, right: Tensor<T>, accumulatingInto: Accumulator.Type) -> Tensor<Accumulator> {
    assert(left.rank == 2 && right.rank == 2, "matrix multiplication is only defined for tensors of rank 2")
    assert(left.dimensions[0] == right.dimensions[1], "incompatible matrix shape")
    
    let K = left.dimensions[0]
    let M = left.dimensions[1]
    let N = right.dimensions[0]
    
    let product =  Tensor<Accumulator>(device: left.mtl_tensor.device, dimensions: [N, M])
    for m in 0..<M {
        for n in 0..<N {
            var accumulator : Accumulator = 0
            for k in 0..<K {
                accumulator += (left[k, m] * right[n, k]) as! Accumulator
            }
            product[n, m] = accumulator
        }
    }
        
    return product
}


struct MMAKernel<Input: TensorData, Output: TensorData>  {
    let M : Int
    let N : Int
    let K : Int
    let pipeline: MTLComputePipelineState
    let ops_per_threadgroup: Int
    let n_threads: Int
    let mtl_env : MetalEnv
    
    var input_type: Input.Type {
        return Input.self
    }
    
    var output_type: Output.Type {
        return Output.self
    }
    
    func makeOutputTensor(amplify factor: Int = 1) -> Tensor<Output> {
        return mtl_env.makeMatrix(withData: Output.self, dimensions: (N*factor, M*factor))
    }
    
    
    func dispatch(
        _ a: Tensor<Input>,
        _ b: Tensor<Input>,
        _ c: Tensor<Output>,
        withTreadgroups n_treadgroups: Int,
        repeat times: Int,
        callback : Optional<(Double, Int) throws -> Void> = nil
    ) throws {
        assert(a.rank == 2 && a.dimensions == [K, M], "tensor a has invalid shape")
        assert(b.rank == 2 && b.dimensions == [N, K], "tensor b has invalid shape")
        assert(c.rank == 2 && c.dimensions[0] >= N && c.dimensions[1] >= M, "tensor b has invalid shape")
                
        let args = mtl_env.makeArgumentTable(maxBindings: 3)
        let resident = mtl_env.makeResidencySet()
        
        resident.addAllocations([a.mtl_tensor, b.mtl_tensor, c.mtl_tensor])
        args.setResource(a.gpuResourceID, bufferIndex: 0)
        args.setResource(b.gpuResourceID, bufferIndex: 1)
        args.setResource(c.gpuResourceID, bufferIndex: 2)
        
        
        for _ in 0..<times {
            let elapsed = try dispatchComputeAndMeasureTime(metal_env) { encoder in
                encoder.commandBuffer!.useResidencySet(resident)
                encoder.setArgumentTable(args)
                encoder.setComputePipelineState(self.pipeline)
                
                encoder.dispatchThreadgroups(
                    threadgroupsPerGrid: MTLSize(width: n_treadgroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: self.n_threads, height: 1, depth: 1)
                )
            }
            
            if let callback = callback {
                try callback(elapsed, ops_per_threadgroup*n_treadgroups)
            }
        }
    }
}



func make_naive_mma_kernel<Input: TensorData, Output: TensorData>(
    _ a: Tensor<Input>,
    _ b: Tensor<Input>,
    accumulatingInto: Output.Type,
    unroll times: Int
) throws -> MMAKernel<Input, Output> {
    assert(a.rank == 2 && b.rank == 2, "matrix multiplication is only defined for tensors of rank 2")
    assert(a.dimensions[0] == b.dimensions[1], "incompatible matrix shape")
    
    let K = a.dimensions[0]
    let M = a.dimensions[1]
    let N = b.dimensions[0]

    let input_type = Input.self.native_type
    let output_type = Output.self.native_type
    
    // naive mma kernel
    let mma_kernel_source = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        using Input  = \(input_type);
        using Output = \(output_type);
    
        
        template<typename T> struct highest_precision;
        template<> struct highest_precision<half> { typedef float type; };
        template<> struct highest_precision<float> { typedef float type; };
        template<> struct highest_precision<int8_t> { typedef int type; };
        template<> struct highest_precision<int> { typedef int type; };
        
        using Accumulator = typename highest_precision<Output>::type;
    
        // fma for integers
        int fma(int a, int b, int c) {
          return a*b + c;
        }
        
        using namespace metal;
        using namespace mpp;

        uint32_t thread_id [[thread_index_in_threadgroup]];
    
        kernel void mma_kernel(
          tensor<device Input, dextents<int, 2>> A,
          tensor<device Input, dextents<int, 2>> B,
          tensor<device Output, dextents<int, 2>> C
        ) {
            int M = A.get_extent(1);
            int K = A.get_extent(0);
            int N = B.get_extent(0);
        

            if (thread_id == 0) {
                for (int i = 0; i < \(times); i++) 
                for (int m = 0; m < M; m ++) {
                    for (int n = 0; n < N; n++) {
                        Accumulator sum = static_cast<Accumulator>(C[n, m]);
                        for (int k = 0; k < K; k++) {
                            sum = fma(static_cast<Accumulator>(A[k, m]), static_cast<Accumulator>(B[n, k]), sum);
                        }
                        C[n, m] = static_cast<Output>(sum);
                    }
                }
            }
        }
    """
    
    let mtl_env = MetalEnv(device: a.mtl_tensor.device)
    let pipeline = try mtl_env.makePipeline(name: "mma_kernel", source: mma_kernel_source)
    

    return MMAKernel(
        M: M,
        N: N,
        K: K,
        pipeline: pipeline,
        ops_per_threadgroup: 2*(K*N*M)*times,
        n_threads: pipeline.threadExecutionWidth,
        mtl_env : mtl_env
    )
}


func make_mma_kernel_global_mem<Input: TensorData, Output: TensorData>(
    _ a: Tensor<Input>,
    _ b: Tensor<Input>,
    accumulatingInto: Output.Type,
    unroll times: Int,
    reduced_precision: Bool = false
) throws -> MMAKernel<Input, Output> {
    assert(a.rank == 2 && b.rank == 2, "matrix multiplication is only defined for tensors of rank 2")
    assert(a.dimensions[0] == b.dimensions[1], "incompatible matrix shape")
    
    let K = a.dimensions[0]
    let M = a.dimensions[1]
    let N = b.dimensions[0]

    let input_type = Input.self.native_type
    let output_type = Output.self.native_type
    
    // naive mma kernel
    
    let mma_kernel_source = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    
        constexpr constant int M = \(M);
        constexpr constant int N = \(N);
        constexpr constant int K = \(K);
    
        using Input  = \(input_type);
        using Output = \(output_type);

        using namespace metal;
        using namespace mpp;

        uint32_t thread_id [[thread_index_in_threadgroup]];
    
        kernel void mma_kernel(
          tensor<device Input, dextents<int, 2>> A,
          tensor<device Input, dextents<int, 2>> B,
          tensor<device Output, dextents<int, 2>> C
        ) {
            constexpr auto mma_descriptor = tensor_ops::matmul2d_descriptor(
                // matrix shape
                M, 
                N,
                K,
                // transpose
                false, 
                false, 
                // reduced precision
                \(reduced_precision ? "true" : "false"),
                // accumulate
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
            );
        
            auto mma_op = tensor_ops::matmul2d<mma_descriptor, execution_simdgroups<1>>();
            
            // perform the operation
            \(Array(repeating: "mma_op.run(A, B, C);", count: times).joined(separator: "\n        "))
        }
    """
      
    let mtl_env = MetalEnv(device: a.mtl_tensor.device)
    let pipeline = try mtl_env.makePipeline(name: "mma_kernel", source: mma_kernel_source)
    
    return MMAKernel(
        M: M,
        N: N,
        K: K,
        pipeline: pipeline,
        ops_per_threadgroup: 2*(K*N*M)*times,
        n_threads: pipeline.threadExecutionWidth,
        mtl_env : mtl_env
    )
}



func make_mma_kernel_cooperative<Input: TensorData, Output: TensorData>(
    _ a: Tensor<Input>,
    _ b: Tensor<Input>,
    accumulatingInto: Output.Type,
    unroll times: Int,
    reduced_precision: Bool = false
) throws -> MMAKernel<Input, Output> {
    assert(a.rank == 2 && b.rank == 2, "matrix multiplication is only defined for tensors of rank 2")
    assert(a.dimensions[0] == b.dimensions[1], "incompatible matrix shape")
    
    let K = a.dimensions[0]
    let M = a.dimensions[1]
    let N = b.dimensions[0]

    let input_type = Input.self.native_type
    let output_type = Output.self.native_type
    
    // naive mma kernel
    let mma_kernel_source = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    
        constexpr constant int M = \(M);
        constexpr constant int N = \(N);
        constexpr constant int K = \(K);
    
        using Input  = \(input_type);
        using Output = \(output_type);

        using namespace metal;
        using namespace mpp;

        uint32_t thread_id [[thread_index_in_threadgroup]];
        uint32_t threadgroup_id [[threadgroup_position_in_grid]];
        
    
        kernel void mma_kernel(
          tensor<device Input, dextents<int, 2>> A,
          tensor<device Input, dextents<int, 2>> B,
          tensor<device Output, dextents<int, 2>> C
        ) {
            constexpr auto mma_descriptor = tensor_ops::matmul2d_descriptor(
                // matrix shape
                M, 
                N,
                K,
                // transpose
                false, 
                false, 
                // reduced precision
                \(reduced_precision ? "true" : "false"),
                // accumulate
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
            );
        
            auto mma_op = tensor_ops::matmul2d<mma_descriptor, execution_simdgroups<1>>();
            
            // init the cooperative tensor
            auto tmp = mma_op.get_destination_cooperative_tensor<decltype(A), decltype(B), Output>();
            for (int i = 0; i < tmp.get_capacity(); i++) {
               if (tmp.is_valid_element(i)) tmp[i] = 0;
               //tmp[i] = 0;
            }
            
            // perform the operation
            \(Array(repeating: "mma_op.run(A, B, tmp);", count: times).joined(separator: "\n        "))
    
            // store the data
            tmp.store(C);
        }
    """
        
    
    let mtl_env = MetalEnv(device: a.mtl_tensor.device)
    let pipeline = try mtl_env.makePipeline(name: "mma_kernel", source: mma_kernel_source)
    

    return MMAKernel(
        M: M,
        N: N,
        K: K,
        pipeline: pipeline,
        ops_per_threadgroup: 2*(K*N*M)*times,
        n_threads: pipeline.threadExecutionWidth,
        mtl_env : mtl_env
    )
}



extension Double {
    func rounded(toPlaces places: Int) -> Double {
        let factor = pow(10.0, Double(places))
        return (self * factor).rounded() / factor
    }
}

enum MMAKernelError : Error {
    case invalidHostTensorOutput(expected: any AnyTensor, observed: any AnyTensor)
    case invalidCooperativeTensorOutput(expected: any AnyTensor, observed: any AnyTensor)
}

func mma_bench<Input: TensorData, Output: TensorData>(
    env: MetalEnv,
    M: Int,
    N: Int,
    K: Int,
    input: Input.Type,
    output: Output.Type,
    repeat times: Int
) throws -> [Double]  {
    // allocate inputs
    let a = env.makeMatrix(withData: Input.self, dimensions: (K, M), init: {i, j in 1})
    let b = env.makeMatrix(withData: Input.self, dimensions: (N, K), init: {i, j in 1})
    
    // loop repetitions to minimize setup overhead
    let unroll = 64
    
    // mma kernels
    // naive single-threaded loops — checking the invariants
    let mma_kernel_naive = try make_naive_mma_kernel(a, b, accumulatingInto: Output.self, unroll: unroll)
    // performance primitives with accumulation directly into the output
    let mma_kernel_accum_host = try  make_mma_kernel_global_mem(a, b, accumulatingInto: Output.self, unroll: unroll)
    // performanc eprimitives with accumulation into a cooperative tensor
    let mma_kernel_accum_thread = try  make_mma_kernel_cooperative(a, b, accumulatingInto: Output.self, unroll: unroll)

    
    // calibrate outputs
    let output0 = mma_kernel_naive.makeOutputTensor()
    let output1 = mma_kernel_accum_host.makeOutputTensor()
    
    // establish baseline output and check ivnariants by running the basic kernels
    // note — we use a single threadgroup to avoid accumulation into host tensor
    try mma_kernel_naive.dispatch(a, b, output0, withTreadgroups: 1, repeat: 1)
    try mma_kernel_accum_host.dispatch(a, b, output1, withTreadgroups: 1, repeat: 1)
    
    let residuals = zip(output0.data, output1.data).map { $0.upcasted - $1.upcasted }
    let error = residuals.reduce(0, { $0 + $1*$1})
    
    // results shold be identical
    if (error.isNaN || error >= 1e-6) {
//        print("mma kernel baselines do not match (error: \(error))")
//        print(output0)
//        print(output1)
        throw MMAKernelError.invalidHostTensorOutput(expected: output0, observed: output1)
    }
    assert(error < 1e-6, "mma kernel baselines do not match (error: \(error))")
    
    
    // run the actual benchmarks
    // modulate the number of threads to always have a comparable amount of ops per invocation
    // 1M threads per core for m8n8k8 seems to be a reasonable effort for benchmarking
    let threads_per_core = 1024*1024*8 / ((M*N*K)/(8*8*8))
    let threadgroups_per_core = threads_per_core / mma_kernel_accum_thread.n_threads
    let n_threadgroups = n_gpu_cores * threadgroups_per_core
        
    var results : [Double] = []
    try mma_kernel_accum_thread.dispatch(a, b, output1, withTreadgroups: n_threadgroups, repeat: times) {elapsed, n_ops in
        let flops = Double(n_ops)/elapsed/1e9
        print("\(flops.rounded(toPlaces: 2)) TOPS")
        results.append(flops)
        
        let residuals = zip(output0.data, output1.data).map { $0.upcasted - $1.upcasted }
        let error = residuals.reduce(0, { $0 + $1*$1})
        
        if (error.isNaN || error >= 1e-6) {
//            print("mma kernel outputs do not match (error: \(error))")
//            print(output0)
//            print(output1)
            throw MMAKernelError.invalidCooperativeTensorOutput(expected: output0, observed: output1)
        }
      
        assert(error < 1e-6, "mma kernel outputs do not match (error: \(error))")
        output1.fill(value: 0)
    }
    
    
    return results
}
