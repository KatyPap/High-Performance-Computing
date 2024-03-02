#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

  // for each given file name/argument...
  for (int i = 1; i < argc; ++i) {
      FILE *file = fopen(argv[i], "rb"); //... open file

      if (file == NULL) {
          printf("Error opening file %s\n", argv[i]);
          return 1;
      }

      double sum = 0.0;
      double value;

      while (fread(&value, sizeof(double), 1, file) == 1) { 
          sum += value;
      }

      printf("Sum of values in %s: %f\n", argv[i], sum);

      fclose(file);
    }

    return 0;
}
