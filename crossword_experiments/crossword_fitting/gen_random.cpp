
#include <iostream>
#include <fstream>
using namespace std;

/** 
 * Program to generate random crosswords based on a layout and grid size.
 * The output of the program is 100k grids and the corresponding words in the grid. 
 */
int main()
{
    int gridSize = 4;
    // Assuming square crosswords for now
    int nCrosswords = 100000;
    // Generate crosswords from layout
    for (int i = 0; i < nCrosswords; ++i) {
        char board[gridSize][gridSize];
        // board[i][j] = 'x' means black square
        // board[0][3] = board[3][0] = 'x';

        ofstream out;
        out.open("gen_crosswords/crossword" + to_string(i) + ".txt");

        // Generate words
        for (int j = 0; j < gridSize; ++j) {
            for (int k = 0; k < gridSize; ++k) {
                if (board[j][k] != 'x') {
                    int n = rand() % 26;
                    char c = (char)(n + 65);
                    board[j][k] = c;
                }
            }
        }

        // Print the crossword
        for (int j = 0; j < gridSize; ++j) {
            for (int k = 0; k < gridSize; ++k) {
                out << board[j][k] << " ";
            }
        }
        out << endl;

        // Extract words
        ofstream words;
        words.open("gen_words/word" + to_string(i) + ".txt");
        // Horizontally
        for (int j = 0; j < gridSize; ++j) {
            string word = "";
            for (int k = 0; k < gridSize; ++k) {
                if (board[j][k] != 'x')
                    word += board[j][k];
                if (board[j][k] == 'x' || k == gridSize - 1) {
                    if (word.size() > 0) {
                        words << word << " " << endl;
                        word = "";
                    }
                    continue;
                }
            }
        }
        // Vertically
        for (int k = 0; k < gridSize; ++k) {
            string word = "";
            for (int j = 0; j < gridSize; ++j) {
                if (board[j][k] != 'x')
                    word += board[j][k];
                if (board[j][k] == 'x' || j == gridSize - 1) {
                    if (word.size() > 0) {
                        words << word << " " << endl;
                        word = "";
                    }
                    continue;
                }
            }
        }
    }
}
