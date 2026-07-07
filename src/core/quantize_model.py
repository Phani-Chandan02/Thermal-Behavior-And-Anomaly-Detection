import os
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("[ERROR] Please install onnxruntime: pip install onnxruntime")
    exit(1)

# ==========================================
# CONFIG & PATHS
# ==========================================
INPUT_MODEL = "best.onnx"
OUTPUT_MODEL = "best_int8.onnx"

def main():
    print(f"[INFO] Starting Dynamic INT8 Quantization for {INPUT_MODEL}...")
    
    if not os.path.exists(INPUT_MODEL):
        print(f"[ERROR] Input model {INPUT_MODEL} does not exist.")
        print("Please ensure you have exported the model to ONNX format first.")
        return

    # Perform dynamic quantization
    try:
        quantize_dynamic(
            model_input=INPUT_MODEL,
            model_output=OUTPUT_MODEL,
            weight_type=QuantType.QInt8
        )
        print("[SUCCESS] Quantization successful!")
        print(f"[INFO] Saved quantized model to: {OUTPUT_MODEL}")
        
        # Compare sizes
        orig_size = os.path.getsize(INPUT_MODEL) / (1024 * 1024)
        quant_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
        
        print("\n[STATS] Model Size Comparison:")
        print(f"  Original (FP32): {orig_size:.2f} MB")
        print(f"  Quantized (INT8): {quant_size:.2f} MB")
        print(f"  Reduction: {((orig_size - quant_size) / orig_size) * 100:.1f}%")
        
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")

if __name__ == "__main__":
    main()
