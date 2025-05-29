import argparse
import numpy as np
import cv2
from improved_iris_recognition import ImprovedIrisRecognition

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ganzin Iris Recognition Challenge')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='Input file to specify a list of sampled pairs')
    parser.add_argument('--output', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the testing results')
    parser.add_argument('--model', type=str, default='best_model.pkl',
                        help='Path to RITNet model weights')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()
    
    # Initialize the improved iris recognition system
    print(" Initializing improved iris recognition system...")
    iris_system = ImprovedIrisRecognition(args.model, debug=args.debug)
    print(" System initialization completed!")

    scores = []
    
    with open(args.input, 'r') as in_file, open(args.output, 'w') as out_file:
        lines = in_file.readlines()
        total_lines = len(lines)
        
        for line_idx, line in enumerate(lines):
            lineparts = line.split(',')
            img1_path = lineparts[0].strip()
            img2_path = lineparts[1].strip()

            # Read images
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                print(f" Unable to read image {img1_path} or {img2_path}")
                score = 0.5  # Default score
            else:
                # Use improved iris recognition algorithm to calculate similarity score
                score = iris_system.compare_iris(img1, img2)

            scores.append(score)
            output_line = f"{img1_path}, {img2_path}, {score:.6f}"
            
            # Display progress every 10%
            if (line_idx + 1) % max(1, total_lines // 10) == 0 or line_idx == 0:
                progress = (line_idx + 1) / total_lines * 100
                current_mean = np.mean(scores)
                current_std = np.std(scores)
                print(f" Progress: {progress:.1f}% ({line_idx+1}/{total_lines}) | "
                      f"Current scores - Mean: {current_mean:.4f}, Std: {current_std:.4f}, "
                      f"Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            
            out_file.write(output_line.rstrip('\n') + '\n')
    
    # Final statistics
    final_mean = np.mean(scores)
    final_std = np.std(scores)
    final_min = np.min(scores)
    final_max = np.max(scores)
    
    print("\n Final Statistics:")
    print(f"   Total comparisons: {len(scores)}")
    print(f"   Score mean: {final_mean:.4f}")
    print(f"   Score std: {final_std:.4f}")
    print(f"   Score range: [{final_min:.4f}, {final_max:.4f}]")
    
    # Analyze score distribution
    low_scores = np.sum(np.array(scores) < 0.3)
    mid_scores = np.sum((np.array(scores) >= 0.3) & (np.array(scores) < 0.7))
    high_scores = np.sum(np.array(scores) >= 0.7)
    
    print(f"   Score distribution:")
    print(f"     Low scores (< 0.3): {low_scores} ({low_scores/len(scores)*100:.1f}%)")
    print(f"     Mid scores (0.3-0.7): {mid_scores} ({mid_scores/len(scores)*100:.1f}%)")
    print(f"     High scores (> 0.7): {high_scores} ({high_scores/len(scores)*100:.1f}%)")
    