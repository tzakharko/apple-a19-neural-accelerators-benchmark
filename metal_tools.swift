import Metal



class MetalEnv {
    let device : MTLDevice
    
    init() {
        self.device = MTLCreateSystemDefaultDevice()!
    }
    
    init(device: MTLDevice) {
        self.device = device
    }
        
    func makeQueue() -> MTL4CommandQueue {
        return device.makeMTL4CommandQueue()!
    }
    
    func makePipeline(name: String) -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!
     
        return try! device.makeComputePipelineState(
            function: library.makeFunction(name: name)!
        )
    }

    
    func makePipeline(name: String, source: String, options: MTLCompileOptions? = nil) throws -> MTLComputePipelineState {
        
        let library = try device.makeLibrary(source: source, options: options)
     
        return try device.makeComputePipelineState(
            function: library.makeFunction(name: name)!
        )
    }


    func makeResidencySet() -> MTLResidencySet {
        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = 5
        
        return try! device.makeResidencySet(descriptor: descriptor)
    }


    func makeArgumentTable(maxBindings n: Int) -> MTL4ArgumentTable {
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = n
        
        return try! device.makeArgumentTable(descriptor: descriptor)
    }


    func makeTimestampCounters() -> MTL4CounterHeap {
        let descriptor = MTL4CounterHeapDescriptor()
        descriptor.type = .timestamp
        descriptor.count = 2
        
        return try! device.makeCounterHeap(descriptor: descriptor)
    }
    
    func makeTensor<T: TensorData>(withData type: T.Type, dimensions: [Int]) -> Tensor<T> {
        return Tensor<T>(device: device, dimensions: dimensions)
    }
    
    func makeMatrix<T: TensorData>(withData type: T.Type, dimensions: (Int, Int), init data: Optional<(Int, Int)->T> = nil) -> Tensor<T> {
        let tensor = makeTensor(withData: type, dimensions: [dimensions.0, dimensions.1])
        guard let data = data else { return tensor }
        
        for i in 0..<dimensions.0 {
            for j in 0..<dimensions.1 {
                tensor[i, j] = data(i, j)
            }
        }
        return tensor
    }

    func resolveTimestampCounters(_ counters: MTL4CounterHeap) -> Double {
        let counter_data = try! counters.resolveCounterRange(0..<2)!

        
        let t0 = counter_data.bytes.unsafeLoad(fromUncheckedByteOffset: 0, as: UInt64.self)
        let t1 = counter_data.bytes.unsafeLoad(fromUncheckedByteOffset: 8, as: UInt64.self)
        
        if (t1 == 0 || t0 == t1)  { return 0.0 }
        
        return Double(t1 - t0)/Double(device.queryTimestampFrequency())
    }
}



func dispatchComputeAndMeasureTime(_ env: MetalEnv, _ encode: (MTL4ComputeCommandEncoder) -> Void) throws -> Double {
    let device = env.device
    let queue = env.makeQueue()
    
    // synchronize GPU work
    var cmd_elapsed = 0.0
    let semaphore = DispatchSemaphore(value: 0)
    let sync_options = MTL4CommitOptions()
    var error : (any Error)? = nil
    sync_options.addFeedbackHandler({ feedback in
        error = feedback.error
        cmd_elapsed = feedback.gpuEndTime - feedback.gpuStartTime
        semaphore.signal()
    })
    
    
    let timestamps = env.makeTimestampCounters()
    let cmd = device.makeCommandBuffer()!
    
    cmd.beginCommandBuffer(allocator: device.makeCommandAllocator()!)
    cmd.writeTimestamp(counterHeap: timestamps, index: 0)
    
    let encoder = cmd.makeComputeCommandEncoder()!
    encode(encoder)
    encoder.endEncoding()
    cmd.writeTimestamp(counterHeap: timestamps, index: 1)
    cmd.endCommandBuffer()
    
    queue.commit([cmd], options: sync_options)
    semaphore.wait()
    
    if let error = error {
        throw error
    }
    
    // resolve the elapsed time, checking for issues
    let counters_elapsed = env.resolveTimestampCounters(timestamps)
    if (abs(counters_elapsed - cmd_elapsed) > 0.05) {
        print("timing mismatch")
        print("elapsed in feedback buffer: \(cmd_elapsed)")
        print("elapsed in timestamp buffer: \(counters_elapsed)")
    }
    
    return counters_elapsed
}

#if os(macOS)
import IOKit

func detectGPUCoreCount() -> Int? {
    guard let matching = IOServiceMatching("AGXAccelerator") else { return nil }
    

    var iterator: io_iterator_t = 0
    guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else {
        return nil
    }
    defer { IOObjectRelease(iterator) }

    var service = IOIteratorNext(iterator)
    while service != 0 {
        var cfProps: Unmanaged<CFMutableDictionary>?
        if IORegistryEntryCreateCFProperties(service, &cfProps, kCFAllocatorDefault, 0) == KERN_SUCCESS,
           let dict = cfProps?.takeRetainedValue() as NSDictionary?,
           let n = dict["gpu-core-count"] as? Int {
            IOObjectRelease(service)
            return n
        }
        IOObjectRelease(service)
        service = IOIteratorNext(iterator)
    }
    return nil
}
#else
func detectGPUCoreCount() -> Int? {
    // detect model
    var info = utsname()
    uname(&info)

    let model = withUnsafePointer(to: &info.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: 1) { String.init(validatingUTF8: $0)! }
    }

    switch model {
        case "iPhone18,3": return 5
        default: return nil
    }
}
#endif
