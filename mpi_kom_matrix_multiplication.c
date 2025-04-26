#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <unistd.h>


#define N 2016
#define PP 2
#define P 4
#define BlockSize (N / PP)
#define K 4

double startwtime1, startwtime2, endwtime;

float errorMargin = 0.01;
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

	char hostname[1024];
    gethostname(hostname, 1024);




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


	if (ncpus != P) {
		if (my_rank == 0) printf("wywolano obliczenia iloczynu macierzy metoda cannona na %d procesach - uruchom mpiexec -n %d matrixmult\n", ncpus, P);
		MPI_Finalize(); 
		exit(0);
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




		for(int process_rank = 0; process_rank < P; process_rank++){
			if(process_rank == 0){
				for(int i = 0; i < BlockSize;i++){
					for(int j = 0; j < BlockSize; j++){
						a[i][j] = A[i][j];
						b[i][j] = B[i][j];
					}
				}
				continue;
			}

			int block_x,block_y;
			block_x = process_rank % PP;
			block_y = process_rank / PP;

			MPI_Datatype subarray_a, subarray_b; 

			//Preparing slice for a
			int starts_a[2] = {block_y * BlockSize, ((block_x + block_y)%PP) * BlockSize};        // starting index of the block
			MPI_Type_create_subarray(2, arrayDims, subarrayDims, starts_a, MPI_ORDER_C, MPI_FLOAT, &subarray_a);
			MPI_Type_commit(&subarray_a);
			MPI_Isend(&A[0][0], 1, subarray_a, process_rank, tag, MPI_COMM_WORLD,&reqSend[0]);
			MPI_Type_free(&subarray_a);

			//Preapring slice for b
			int starts_b[2] = {((block_y+block_x)%PP) * BlockSize, block_x * BlockSize};        // starting index of the block
			MPI_Type_create_subarray(2, arrayDims, subarrayDims, starts_b, MPI_ORDER_C, MPI_FLOAT, &subarray_b);
			MPI_Type_commit(&subarray_b);
			MPI_Isend(&B[0][0], 1, subarray_b, process_rank, tag, MPI_COMM_WORLD,&reqSend[1]);
			MPI_Type_free(&subarray_b);


		}
	}
	else
	{
		MPI_Irecv(a, BlockSize * BlockSize, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, reqRecv);
		//test konca komunikacji
		MPI_Wait(reqRecv,statRecv);

		MPI_Irecv(b, BlockSize * BlockSize, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &reqRecv[1]);
		//test konca komunikacji
		MPI_Wait(&reqRecv[1],&statRecv[1]);

	}

	// if(my_rank != 0) {
	// 	MPI_Recv(&tag, 1, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, &status);
	// }
	
	// Can print subarray of each process for debug
	// printf("my rank is: %d\n", my_rank);
	// printf("This is my a\n");
	// for (int i = 0; i < BlockSize; i++){
	// 	for (int j = 0; j < BlockSize; j++)
	// 	{
	// 		printf("%6.1f",a[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("This is my b\n");
	// for (int i = 0; i < BlockSize; i++){
	// 	for (int j = 0; j < BlockSize; j++)
	// 	{
	// 		printf("%6.1f",b[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// if(my_rank != P-1) MPI_Send(&tag,1,MPI_INT,my_rank+1,tag,MPI_COMM_WORLD);


	//przygotowanie lokalnej  tablicy wynikowej 
	// row = my_rank / PP; col = my_rank % PP;
	int x,y;
	x = my_rank % PP; //row
	y = my_rank / PP; //column
	int down_neighbor = ((y+1)%PP) * PP + x;
	int up_neighbor = ((y-1+PP)%PP) * PP + x;
	int right_neighbor = y * PP + ((x + 1)%PP);
	int left_neighbor = y * PP + ((x - 1 + PP)%PP);

	for (int i = 0; i < BlockSize; i++)
		for (int j = 0; j < BlockSize; j++)
		{
			c[i][j] = 0;

		}
	printf("Process %d on %s is commencing calclulations ...\n", my_rank, hostname);

	if (my_rank == 0) startwtime2 = MPI_Wtime();//czas w sekundach

	pra = aa; 
	prb = bb;
	psa = a; 
	psb = b;

	for (int kk = 0; kk < PP; kk++) // KOLENA ITERACJA PRZETWARZANIA (OBLICZENIA I KOMUNIKACJA) 
	{
		//OBLICZENIA
		for (int i = 0; i < BlockSize; i++)
			for (int k = 0; k < BlockSize; k++)
				for (int j = 0; j < BlockSize; j++) 
					c[i][j] += psa[i][k] * psb[k][j];
		//KOMUNIKAJCA MPI_Irecv(adres, ile_słów, typ_danych, odbiorca/nadawca, znacznik, zakres_procesów, Id_komunikacji);
		MPI_Irecv(pra, BlockSize * BlockSize, MPI_FLOAT, right_neighbor, tag, MPI_COMM_WORLD, reqRecv);
		MPI_Irecv(prb, BlockSize * BlockSize, MPI_FLOAT, down_neighbor, tag, MPI_COMM_WORLD, &reqRecv[1]);
		MPI_Isend(psa, BlockSize * BlockSize, MPI_FLOAT, left_neighbor, tag, MPI_COMM_WORLD, reqSend);
		MPI_Isend(psb, BlockSize * BlockSize, MPI_FLOAT, up_neighbor, tag, MPI_COMM_WORLD, &reqSend[1]);
		//OCZEKIWANIE NA KOMUNIKACJĘ ASYNCHRONICZNĄ MPI_Wait(reqRecv, statRecv);
		MPI_Wait(&reqRecv[0], &statRecv[0]);
		MPI_Wait(&reqRecv[1], &statRecv[1]);
		//ZMIANA OBSZARÓW DANYCH LICZONYCH I PRZESYAŁANYCH 
		if (mod = ((mod + 1) % 2)) { pra = a; prb = b; psa = aa; psb = bb;} 
		else { pra = aa; prb = bb; psa = a; psb = b; } 

	}


	printf("Process %d on %s has finished calclulations.\n", my_rank, hostname);


	if (my_rank == 0)
	{
		endwtime = MPI_Wtime();
		printf("Calkowity czas przetwarzania wynosi %f sekund\n", endwtime - startwtime1);
		printf("Calkowity czas obliczen wynosi %f sekund\n", endwtime - startwtime2);

	}

	// test poprawnosci wyniku 
	



	if (my_rank == 0)
	{
		// obliczenia sekwencyjne mnożenia tablic CSek=A*B

		printf("Commencing sequential matrix multiplication ...\n");
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				Cglob[i][j] = 0;

			}
		double startSekTime = MPI_Wtime();

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				CSek[i][j] = 0;

			}


		double startSekTime2 = MPI_Wtime();

		for (int i = 0; i < N; i++)
			for (int k = 0; k < N; k++)
				for (int j = 0; j < N; j++) 
					CSek[i][j] += A[i][k] * B[k][j];
		double endSekTime = MPI_Wtime();
		printf("Sequential matrix multiplication finished in %f seconds.\n\n",endSekTime - startSekTime);

		// odbiór wyników obliczeń równoległych do globalnej tablicy wynikowej Cglob
		printf("Gathering final blocks from other processes ...\n");
		MPI_Datatype block;
		MPI_Request finalReqRecv[P-1];
		MPI_Status  finalStatRecv[P-1];
        if(P != 1) MPI_Send(&tag, 1, MPI_INT, my_rank + 1, tag, MPI_COMM_WORLD);

		for(int process_rank = 0; process_rank < P; process_rank++){

			if(process_rank == 0){

				for(int i = 0; i < BlockSize;i++){
					for(int j = 0; j < BlockSize; j++){
						Cglob[i][j] = c[i][j];
					}
				}
				continue;
			}
            int block_x,block_y;
			block_x = process_rank % PP;
			block_y = process_rank / PP;
            for(int i = 0; i < BlockSize; i++){
                for(int j = 0; j < BlockSize; j++){
                    MPI_Recv(&Cglob[i + (block_y * BlockSize)][(block_x * BlockSize)], BlockSize, MPI_FLOAT, process_rank, tag, MPI_COMM_WORLD, &status);
                }
            }

			
		}

		// MPI_Waitall(P-1, &finalReqRecv[0], &finalStatRecv[0]);
		printf("Gathering final blocks from other processes has been finished\n\n");



		// porównanie poprawności obliczeń (Csek, Cglob) przy uwzględniniu progu poprawności 
		printf("Comparing the results ...\n");
		float correctPositions = 0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				// float error = fabsf(CSek[i][j] - Cglob[i][j]) / (fabsf(Cglob[i][j]) + FLT_EPSILON);
				if(fabs(CSek[i][j]-Cglob[i][j]) < errorMargin * fabs(CSek[i][j])) correctPositions++;
				// else printf("Error on (%d,%d)\nCSek: %f\nCglob: %f\n",i,j,CSek[i][j],Cglob[i][j]);
			}
		}
		printf("Comparing results finished.\n\n");
		printf("Wynik działania algorytmu jest poprawny w %3.4f%%\n", correctPositions * 100 / (N*N));
		// printAB(CSek,Cglob,"CSek", "Cglob");
		double processingTimeCannon = endwtime - startwtime1;
		double calculationTimeCannon = endwtime - startwtime2; 
		printf("Calkowity czas przetwarzania wynosi %f sekund\n", processingTimeCannon);
		printf("Calkowity czas obliczen wynosi %f sekund\n", calculationTimeCannon);
		double processingTimeSequential = endSekTime - startSekTime;
		double calculationTimeSequential = endSekTime - startSekTime2;
		printf("Calkowity czas przetwarzania dla sek wynosi %f sekund\n", processingTimeSequential);
		printf("Calkowity czas obliczen wynosi dla sek %f sekund\n",calculationTimeSequential);
		printf("Prędkość dla metody Cannon to: %f \n", (pow(BlockSize, 3)*PP*P)/calculationTimeCannon);
		printf("Prędkość dla metody sekwencyjnej to: %f \n", pow(N, 3)/calculationTimeSequential);

		printf("Przyspieszenie: %f \n", processingTimeSequential/processingTimeCannon);
		printf("Efektywność obliczeń: %f \n",  (processingTimeSequential/processingTimeCannon) / ((P/K <= 8) ? P : K*8) );
		printf("Względny koszt zrównoleglenia: %f %%\n", ((P*calculationTimeCannon-calculationTimeSequential)/calculationTimeSequential) * 100);
	} else {
		// Send data to root
        MPI_Recv(&tag, 1, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, &status);
        for(int i = 0; i < BlockSize; i++){
                MPI_Send(&c[i][0], BlockSize, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
        }
		if(my_rank != P-1){
            MPI_Send(&tag, 1, MPI_INT, my_rank + 1, tag, MPI_COMM_WORLD);
        }
	}

	// free(A); free(B); free(Cglob); free(CSek);
	// free(a); free(b); free(c);
	// free(aa); free(bb);

	MPI_Finalize();
	return 0;
}
