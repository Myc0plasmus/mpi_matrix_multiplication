#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2016
#define PP 2
#define P 4
#define BlockSize (N / PP)

double startwtime1, startwtime2, endwtime;

float A[N][N], B[N][N], Cglob[N][N], CSek[N][N];
float a[BlockSize][BlockSize], b[BlockSize][BlockSize], c[BlockSize][BlockSize];
float aa[BlockSize][BlockSize], bb[BlockSize][BlockSize];
float(*psa)[BlockSize], (*psb)[BlockSize], (*pra)[BlockSize], (*prb)[BlockSize];

void printAB(float (*A)[N], float (*B)[N],char * first, char * second){
	for (int i = 0; i < N*2; i++){
		if( i == 0 ) printf("%s:\n",first);
		else if( i % N == 0) printf("%s:\n",second);

		for (int j = 0; j < N; j++) {
			printf("%6.1f", (i / N == 0) ? A[i][j] : B[i%N][j]);
		}
		printf("\n");
	}
}

int main(int argc, char** argv)
{
	// float (*A)[N] = malloc(N * sizeof(*A));
	// float (*B)[N] = malloc(N * sizeof(*B));
	// float (*Cglob)[N] = malloc(N * sizeof(*Cglob));
	// float (*CSek)[N] = malloc(N * sizeof(*CSek));
	//
	// float (*a)[BlockSize] = malloc(BlockSize * sizeof(*a));
	// float (*b)[BlockSize] = malloc(BlockSize * sizeof(*b));
	// float (*c)[BlockSize] = malloc(BlockSize * sizeof(*c));
	//
	// float (*aa)[BlockSize] = malloc(BlockSize * sizeof(*aa));
	// float (*bb)[BlockSize] = malloc(BlockSize * sizeof(*bb));

	FILE* plik;
	FILE* plik_out;

	int my_rank, ncpus;
	int row, col, mod = 0;
	int data_received = -1;
	int tag = 101;
	int koniec;

	MPI_Status  statRecv[2];
	MPI_Request reqSend[2], reqRecv[2];
	MPI_Status status;

	MPI_Init(0, 0);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ncpus);


	int arrayDims[2] = {N, N};         // full array size
	int subarrayDims[2] = {BlockSize, BlockSize};      // size of the block to send

	

	
	if (my_rank == 0)
		printf("obliczenia metod  Cannona dla tablicy %d x %d element w \n", N, N);

	if (my_rank == 0) startwtime1 = MPI_Wtime();//czas w sekundach

	//wczytanie danych przez proces rank=0
	if (my_rank == 0)
	{
		plik = fopen("liczby.txt", "r");
		if (plik == NULL)
		{
			printf("Blad otwarcia pliku \"liczby.txt\"\n");
			koniec = 1;
			MPI_Bcast(&koniec, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Finalize();
			exit(0);
		}
		else {
			koniec = 0;
			MPI_Bcast(&koniec, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Bcast(&koniec, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (koniec) { MPI_Finalize(); exit(0); }
	}


	if (my_rank == 0)
	{
		//odczyt danych wejściowych tablica A
		//rozesłanie tablicy a zgodnie z dystrybucją początkową tablicy A
        //odczyt danych wejściowych tablica B
		//rozesłanie tablicy a zgodnie z dystrybucją początkową tablicy B
		//do uzupełnienia

		for (int i = 0; i < N*2; i++) {
			for (int j = 0; j < N; j++) {
				if (fscanf(plik, "%f", (i / N == 0) ? &A[i][j] : &B[i%N][j]) != 1) {
					printf("Blad odczytu floata w [%d][%d]\n", i, j);
					fclose(plik);
					return 1;
				}
			}
		}

		//printAB(A, B, "A", "B");	




	}

	



	if (my_rank == 0)
	{
		// obliczenia sekwencyjne mnożenia tablic CSek=A*B

		printf("Commencing IKJ sequentialmatrix multiplication ...\n");
		double startSekTime = MPI_Wtime();

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				CSek[i][j] = 0;

			}

		for (int i = 0; i < N; i++)
			for (int k = 0; k < N; k++)
				for (int j = 0; j < N; j++) 
					CSek[i][j] += A[i][k] * B[k][j];
		double endSekTime = MPI_Wtime();
		printf("Sequential IKJ matrix multiplication finished in %f seconds\n\n",endSekTime - startSekTime);

		printf("Commencing IJK sequential matrix multiplication.\n");
		double startSekTime2 = MPI_Wtime();

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				CSek[i][j] = 0;

			}

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) 
				for (int k = 0; k < N; k++)
					CSek[i][j] += A[i][k] * B[k][j];
		double endSekTime2 = MPI_Wtime();
		printf("Sequential IJK matrix multiplication finished in %f seconds.\n\n",endSekTime2 - startSekTime2);
	}

	// free(A); free(B); free(Cglob); free(CSek);
	// free(a); free(b); free(c);
	// free(aa); free(bb);

	MPI_Finalize();
	return 0;
}
