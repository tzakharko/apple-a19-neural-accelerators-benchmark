import Metal
import TabularData

// utility to save the csv to the app's documents folder
extension DataFrame {
    func writeCSV(toDocuments name: String) throws -> URL {
        let documents_root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let url = documents_root.appendingPathComponent(name)

        try writeCSV(to: url)
        
        return url
    }
}

let n_gpu_cores = detectGPUCoreCount()!
print("Detected \(n_gpu_cores) GPU cores")

let metal_env = MetalEnv()

// mma benchmarks
do {
    var results = DataFrame()
    results.append(column: Column<String>(name: "input", capacity: 0))
    results.append(column: Column<String>(name: "output", capacity: 0))
    results.append(column: Column<Int>(name: "M", capacity: 0))
    results.append(column: Column<Int>(name: "N", capacity: 0))
    results.append(column: Column<Int>(name: "K", capacity: 0))
    results.append(column: Column<Double>(name: "tops", capacity: 0))

    // type combinations to run
    let input_output : [(any TensorData.Type, any TensorData.Type)] = [
        (Float16.self, Float16.self),
        (Float16.self, Float32.self),
        // (Float32.self, Float32.self),
        (Int8.self, Int32.self)
    ]
    
    // matrix shapes to run
    let shapes = [8, 16, 32].flatMap {m in
        [8, 16, 32].flatMap {n in
            [8, 16, 32, 64, 128].map {k in
                (m, n, k)
            }
        }
    }
        
    for (in_type, out_type) in input_output {
        for (m, n, k) in shapes {
            let fn = "mma_m\(m)n\(n)k\(k).\(in_type.label).\(out_type.label)"
            
            
            print("=== \(fn)")
            do {
                let tops = try mma_bench(env: metal_env, M: m, N: n, K: k, input: in_type, output: out_type, repeat: 30)
                tops.forEach { tops in
                    results.append(row: in_type.label, out_type.label, m, n, k, tops)
                }
            } catch MMAKernelError.invalidHostTensorOutput(let expected, let observed) {
                print("! error: invalid output for the MMA host-bound tensor kernel")
                print("! expected")
                print(expected)
                print("! observed")
                print(observed)
                
                results.append(row: in_type.label, out_type.label, m, n, k, Double.nan)
            } catch MMAKernelError.invalidCooperativeTensorOutput(let expected, let observed) {
                print("! error: invalid output for the MMA cooperative tensor kernel")
                results.append(row: in_type.label, out_type.label, m, n, k, Double.nan)
                
                print("! expected")
                print(expected)
                print("! observed")
                print(observed)
            } catch {
                print("! error: Metal error")
                print(error)
                
                results.append(row: in_type.label, out_type.label, m, n, k, Double.nan)
            }
            
            let _ = try! results.writeCSV(toDocuments: "mma.csv")
        }
    }
    
    let url = try! results.writeCSV(toDocuments: "mma.csv")
    print("=== written results to \(url)")
}
