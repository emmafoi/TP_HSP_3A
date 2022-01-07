#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib> //pour le rand

/* on répertorie comme dans un fichier h*/
void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
__host__ void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
void MatrixMult(float *M1, float *M2, float *Mout, int n);
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n);

void MatrixInit(float *M, int n, int p){
    for (int i = 0; i < n*p; i++){
        M[i] = (rand() % 3) -1;
    }
}

void MatrixPrint(float *M, int n, int p){
    for (int i = 0; i < n*p ; i++){
        if((i+1)%n ==0){
            printf("%f\n",M[i]);
        }else{
            printf("%f ",M[i]);
        }
    }
}

__host__ void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n*p; i++){
        Mout[i] = M1[i] + M2[i];
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int col = blockIdx.x; //premier élément dans l'appel à la fonction
    int row = threadIdx.x; //deuxième élément dans l'appel à la fonction
    Mout[row*n + col] = M1[row*n + col] + M2[row*n + col];
    
    printf("Hello\n");
    //pour lancer le thread, on fait cudaMatrixAdd<<<n,p>>>(M1,M2,Mout,n,p)
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    for(int row = 0; row < n; row++){
        for(int col = 0; col < n; col++){
            int tmp = 0;
            for( int i = 0; i < n; i++){
                tmp += M1[row * n + i] * M2[i * n + col];
            }
            //Write back the result
            Mout[row * n + col] = tmp;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    int col = blockIdx.x; //premier élément dans l'appel à la fonction
    int row = threadIdx.x; //deuxième élément dans l'appel à la fonction
    if(row < n && col < n){
        //Accumulate a partial result
        int tmp = 0;
        for( int i = 0; i < n; i++){
            tmp += M1[row * n + i] * M2[i * n + col];
        }
        //Write back the result
        Mout[row * n + col] = tmp;
    }
}


int main(){
    
    //tailles des matrices
    int n = 2;
    int p = 3;
    
    //taille en mémoire des matrices pour l'addition
    const int ARRAY_SIZE = n*p;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    
    //taille en mémoire des matrices pour la multiplication
    const int ARRAY_SIZE2 = n*n;
    const int ARRAY_BYTES2 = ARRAY_SIZE2 * sizeof(float);
    
    // variables host + allocation de mémoire sur CPU
    // pour l'addition
    float *a, *b, *out;
    a   = (float*)malloc(ARRAY_BYTES);
    b   = (float*)malloc(ARRAY_BYTES);
    out = (float*)malloc(ARRAY_BYTES);
    //pour la multiplication
    float *a2, *b2, *out2;
    a2   = (float*)malloc(ARRAY_BYTES2);
    b2   = (float*)malloc(ARRAY_BYTES2);
    out2 = (float*)malloc(ARRAY_BYTES2);
    
    //initialisation des matrices tests:
    //pour l'addition
    MatrixInit(a, n, p);
    MatrixInit(b, n, p);
    //pour la multiplication
    MatrixInit(a2, n, n);
    MatrixInit(b2, n, n);
    
    //calcul sur cpu :
    //addition
    MatrixAdd(a, b, out, n, p);
    //multiplication
    MatrixMult(a2, b2, out2, n);
    
    //affichage :
    printf("Matrice a :\n");
    MatrixPrint(a, n, p);
    printf("Matrice b :\n");
    MatrixPrint(b, n, p);
    printf("Matrice a+b sur cpu :\n");
    MatrixPrint(out, n, p);
    printf("Matrice a2 :\n");
    MatrixPrint(a2, n, n);
    printf("Matrice b2 :\n");
    MatrixPrint(b2, n, n);
    printf("Matrice a2*b2 sur cpu :\n");
    MatrixPrint(out2, n, n);
    
    //calcul GPU
    // variables device + allocation de mémoire sur GPU
    float *d_a, *d_b, *d_out, *out1, *d_a2, *d_b2, *d_out2, *out22;
    cudaMalloc((void **) &d_a, ARRAY_BYTES);
    cudaMalloc((void **) &d_b, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);
    cudaMalloc((void **) &d_a2, ARRAY_BYTES2);
    cudaMalloc((void **) &d_b2, ARRAY_BYTES2);
    cudaMalloc((void **) &d_out2, ARRAY_BYTES2);
    out1 = (float*)malloc(ARRAY_BYTES);
    out22 = (float*)malloc(ARRAY_BYTES2);
    
    //transfert de données CPU vers GPU pour le calcul sur gpu
    cudaMemcpy(d_a, a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a2, a2, ARRAY_BYTES2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, ARRAY_BYTES2, cudaMemcpyHostToDevice);
    
    //calcul sur gpu
    //addition : on fait n blocks de p threads
    printf("Ici\n");
    cudaMatrixAdd<<<n,p>>>(d_a,d_b,d_out,n,p);
    //multiplication : on fait n blocks de n threads
    cudaMatrixMult<<<n,n>>>(d_a2,d_b2,d_out2,n);
    
    //récupération des données du GPU vers le CPU
    cudaMemcpy(out1, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(out22, d_out2, ARRAY_BYTES2, cudaMemcpyDeviceToHost);
    
    //affichage du résultat
    printf("Matrice a+b sur gpu :\n");
    MatrixPrint(out1, n, p);
    printf("Matrice a2*b2 sur gpu :\n");
    MatrixPrint(out22, n, n);
    
    //libération des ressources 
    // sur GPU :
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_a2);
    cudaFree(d_b2);
    cudaFree(d_out2);
    // sur CPU
    free(a);
    free(b);
    free(a2);
    free(b2);
    free(out);
    free(out1);
    free(out2);
    free(out22);
    
    // This call waits for all of the submitted GPU work to complete
    cudaDeviceSynchronize();
    
    return 0;
}
