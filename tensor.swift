import Metal


protocol TensorData : Numeric {
    static var mtlType: MTLTensorDataType { get }
    static var label: String { get }
    static var native_type: String { get }
    
    var upcasted: Double { get }
}
extension Float32: TensorData {
    static  var mtlType: MTLTensorDataType { .float32 }
    static  var label: String { "f32" }
    static  var native_type: String { "float" }
    
    var upcasted: Double { Double(self) }
}
extension Float16: TensorData {
    static var mtlType: MTLTensorDataType { .float16 }
    static  var label: String { "f16" }
    static  var native_type: String { "half" }
    
    var upcasted: Double { Double(self) }
}
extension Int8: TensorData {
    static  var mtlType: MTLTensorDataType { .int8 }
    static  var label: String { "i8" }
    static  var native_type: String { "int8_t" }
    
    var upcasted: Double { Double(self) }
}
extension Int32: TensorData {
    static  var mtlType: MTLTensorDataType { .int32 }
    static  var label: String { "i32" }
    static  var native_type: String { "int" }
    
    var upcasted: Double { Double(self) }
}

protocol AnyTensor : CustomStringConvertible{
    var mtl_tensor: MTLTensor { get }
    
    var gpuResourceID: MTLResourceID { get }
    var data_type: any TensorData.Type { get }
}



class Tensor<T:TensorData> : AnyTensor {
    var mtl_tensor: MTLTensor;
    
    init(device: MTLDevice, dimensions: [Int]) {
        let descriptor = MTLTensorDescriptor()
        
        descriptor.dimensions = MTLTensorExtents(dimensions)!
        descriptor.dataType = T.self.mtlType

        self.mtl_tensor = try! device.makeTensor(descriptor: descriptor)
    }
    
    convenience init(device: MTLDevice, dimensions: [Int], withData: [T]) {
        self.init(device: device, dimensions: dimensions)
        self.data = data
    }
    
    func fill(value: T) {
        let data = Array(repeating: value, count: self.element_count)
        self.data = data
    }

    
    var gpuResourceID: MTLResourceID {
        return mtl_tensor.gpuResourceID
    }
    
    var data_type: any TensorData.Type {
        return T.self
    }
    
    var rank: Int {
        return mtl_tensor.dimensions.rank
    }
    
    var element_count: Int {
        return dimensions.reduce(1, { $0*$1 })
    }
    
    var dimensions: [Int] {
        return mtl_tensor.dimensions.extents
    }
    
    var strides: [Int] {
        var strides = [1]
        var last_stride = 1
        for dim in self.dimensions.dropLast() {
            last_stride *= dim
            strides.append(last_stride)
        }
        return strides
    }
    
    var data: [T] {
        get {
            var data : [T] = .init(repeating: 0, count: self.element_count)
        
            data.withUnsafeMutableBytes { ptr in
                mtl_tensor.getBytes(ptr.baseAddress!,
                                    strides: MTLTensorExtents(self.strides)!,
                                    sliceOrigin: MTLTensorExtents(Array(repeating: 0, count: self.rank))!,
                                    sliceDimensions: mtl_tensor.dimensions)
                
            }
            
            
            //
            //        data.withUnsafeMutableBufferPointer {ptr in
            //            mtl_tensor.getBytes(UnsafeMutableRawPointer(ptr.baseAddress)!, strides: MTLTensorExtents([1, dim0])!, sliceOrigin: MTLTensorExtents([0, 0])!, sliceDimensions: MTLTensorExtents([dim0, dim1])!)
            
            return data
        }
        set(data) {
            assert(data.count == self.element_count, "invalid data size")

            data.withUnsafeBytes { ptr in
                mtl_tensor.replace(
                    sliceOrigin: MTLTensorExtents(Array(repeating: 0, count: self.rank))!,
                    sliceDimensions: mtl_tensor.dimensions,
                    withBytes: ptr.baseAddress!,
                    strides: MTLTensorExtents(self.strides)!
                )
            }
        }
    }
    
    
    subscript(_ indices: Int...) -> T {
        get {
            assert(indices.count == self.rank, "invalid index dimension (got \(indices.count), expected \(self.rank))")
            
            var data: [T] = [0]
            
            data.withUnsafeMutableBytes { ptr in
                mtl_tensor.getBytes(ptr.baseAddress!,
                                    strides: MTLTensorExtents(self.strides)!,
                                    sliceOrigin: MTLTensorExtents(indices as [Int])!,
                                    sliceDimensions: MTLTensorExtents(Array(repeating: 1, count: self.rank))!
                )
            }
            
            return data[0]
        }
        set(value) {
            assert(indices.count == self.rank, "invalid index dimension (got \(indices.count), expected \(self.rank))")
            
            let data = [value]
            
            data.withUnsafeBytes { ptr in
                mtl_tensor.replace(
                    sliceOrigin: MTLTensorExtents(indices as [Int])!,
                    sliceDimensions: MTLTensorExtents(Array(repeating: 1, count: self.rank))!,
                    withBytes: ptr.baseAddress!,
                    strides: MTLTensorExtents(self.strides)!
                )
            }
            
        }
    }
    
    
    var description: String {
        let data = self.data
        let strides = Array(self.strides.dropFirst(2))
        let indent  = Array((0..<self.rank).map( { String(repeating: " ", count: $0*2) } ).reversed())
        
        var out = "tensor<\(T.label)>(\(self.dimensions)) ["
        
        // print dim0 slices
        var delim = "\n"
        let dim0 = self.dimensions[0]
        for i in 0..<(self.element_count/dim0) {
            let start = dim0*i
            let end = dim0*(i+1)
            var delim1 = (start == 0) ? "\n" : ",\n"
            for (i, stride) in strides.enumerated().reversed() {
                if (start % stride == 0) {
                    out += delim1 + indent[i + 1] + "["
                    delim1 = "\n"
                    delim = "\n"
                }
            }

            let slice = data[start..<end]
            out +=  delim + indent[0] + slice.description
            delim = ",\n"
            // write out the delimiters
            for (i, stride) in strides.enumerated() {
                if (end % stride == 0) {
                    out += "\n" + indent[i + 1] + "]"
                }
            }
        }
        
        
        return out + "\n]"
    }
}

func -<T: TensorData> (left: [T], right: [T]) -> [T] {
    assert(left.count == right.count, "difference is only defined for equal-length sequences")
    return Array(zip(left, right).map(-))
}






