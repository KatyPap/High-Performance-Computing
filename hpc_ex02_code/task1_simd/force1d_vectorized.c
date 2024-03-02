#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <immintrin.h>

double get_wtime(void){
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

const size_t N = 1 << 16;
const float eps = 5.0;
const float rm = 0.1;

float compute_force(float *positions, float x0){
  const __m256 eps_v = _mm256_set1_ps(eps);
  const __m256 rm_v = _mm256_set1_ps(rm);
  const __m256 x0_v = _mm256_set1_ps(x0);
  const __m256 rm2_v = _mm256_set1_ps(rm * rm);

  __m256 total_force_v = _mm256_setzero_ps();

  for (size_t i=0; i<N; i+=8){

    __m256 positions_v = _mm256_loadu_ps(&positions[i]);

    __m256 r_v = _mm256_sub_ps(x0_v, positions_v);
    __m256 r2_v = _mm256_mul_ps(r_v, r_v);
    __m256 s2_v = _mm256_div_ps(rm2_v, r2_v);
    __m256 s6_v = _mm256_mul_ps(_mm256_mul_ps(s2_v, s2_v), s2_v);

    __m256 term1_v = _mm256_sub_ps(_mm256_mul_ps(s6_v, s6_v), s6_v);
    __m256 term2_v = _mm256_mul_ps(_mm256_set1_ps(12.0f), _mm256_mul_ps(eps_v, term1_v));
    __m256 force_v = _mm256_div_ps(term2_v, r_v);

    total_force_v = _mm256_add_ps(total_force_v, force_v);
  }
  
  float result[8];
  _mm256_storeu_ps(result, total_force_v);
  float force = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];

  return force;
}


int main(int argc, const char **argv){
  /// init random number generator
  srand48(1);

  float *positions = (float *)aligned_alloc(64, N * sizeof(float));

  for (size_t i = 0; i < N; i++)
    positions[i] = drand48() + 0.1;

  /// timings
  double start, end;

  float x0[] = {0., -0.1, -0.2};
  float f0[] = {0, 0, 0};

  const size_t repetitions = 1000;
  start = get_wtime();
  
  for (size_t i = 0; i < repetitions; ++i){
    for (size_t j = 0; j < 3; ++j)
      f0[j] += compute_force(&positions[0], x0[j]);
  }
  end = get_wtime();

  for (size_t j = 0; j < 3; ++j)
    printf("Force acting at x_0=%lf : %lf\n", x0[j], f0[j] / repetitions);

  printf("elapsed time: %lf mus\n", 1e6 * (end - start));
  free(positions);
  return 0;
}