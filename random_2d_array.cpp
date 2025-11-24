#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// Function to fill a 2D vector with random floats between min and max
std::vector<std::vector<float>> createRandom2DArray(int rows, int cols, float min = 1.0f, float max = 5.0f) {
    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    // Create and fill the 2D vector
    std::vector<std::vector<float>> array(rows, std::vector<float>(cols));
    for (auto& row : array) {
        for (auto& element : row) {
            element = dist(gen);
        }
    }
    return array;
}

// Function to print the 2D array
void print2DArray(const std::vector<std::vector<float>>& array) {
    for (const auto& row : array) {
        for (float val : row) {
            std::cout << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Example usage
    int rows = 3;
    int cols = 4;
    
    std::cout << "Generating a " << rows << "x" << cols 
              << " array with random floats between 1.0 and 5.0:\n";
              
    auto randomArray = createRandom2DArray(rows, cols);
    print2DArray(randomArray);
    
    return 0;
}
