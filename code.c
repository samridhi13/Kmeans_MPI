#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<mpi.h>

int totalProcesses;
int dataPointsPerProcess;
int numClusters;
int dataDim;
int totalDataPoints;

/*double **allocate_memory(int row,int col)
{
  // flattening the 2d array into 1d array
  double *data = (double *)malloc(row * col * sizeof(double));// making flat 1d array
  double **arr = (double **)malloc(row * sizeof(double *));// making an array of pointers where each pointer will point to row in 1d array
  for (int i = 0; i < row;i++)
  {
    arr[i] = &(data[i * col]);
  }
  return arr;
}*/

void Calculatedist(double *recv, double *centroids, double *sum, int dim, int N, int clusters, int *count)
{
  for (int i = 0; i < N; i++)
  {
    double min_res = INFINITY;
    double min_i = -1;
    for (int k = 0; k < clusters;k++)
    {
      double total = 0.0;
      for (int l = 0; l < dim;l++)
      {
        total += pow(abs(recv[i * dim + l] - centroids[k * dim + l]), 2);
      }
      total = srt(total);
      if(total<min_res)
      {
        min_i = k;
        min_res = total;
      }
      
    }
    for (int l = 0; l < dim;l++)
    {
      sum[min_i * dim + l] += recv[i * dim + l];
    }
    count[min_i] += 1;
  }
}

int main(int argc, char* argv[])
{ 
  // Step 1: Initializing the MPI frameowrk.
  int mpirank;
  int iter=100;
  char file_name[100];
  MPI_Init(&argc, &argv); // setting system
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);        // Gives infomration regarding the rank of the process
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses); // Give the Number of Processes
  double *recv=NULL;
  double *data=NULL;
  double *centroids = NULL;
  double *sum = NULL;
  int *totalcount = NULL;
  double *res_sum = NULL;
  int *res_count = NULL;
  int *k_assignment = NULL;
  int *final_assignment = NULL;
  if (mpirank==0)
  {
    // Step2- reads the arguments and data set
    // Step3- Scatter the dataset to all the processes
    sscanf(argv[1], "%d", &dataDim);
    sscanf(argv[2], "%d", &totalDataPoints);
    sscanf(argv[3], "%d", &numClusters);
    dataPointsPerProcess = totalDataPoints / totalProcesses;
    MPI_Bcast(&numClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&totalDataPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dataDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    strcpy(file_name, argv[4]);
    //double **arr;
    //arr = allocate_memory(totalDataPoints, dataDim);
    data = (double *)malloc(totalDataPoints * dataDim * sizeof(double));
    FILE *fp = fopen(file_name, "r");
    char line[100];
    int row = 0;
    while(  fgets(line,sizeof(line),fp))
    {
      char *token = strtok(line, ",");
      int col = 0;
      for (int i = 0; i < dataDim;i++)
      {
        data[row*dataDim+i] = atof(token);
        if(i<dataDim-1)
        {
          token = strtok(NULL, ",");
        }
      }
      row++;
    }
    //selecting the random clusters
    centroids = (*double)malloc(numClusters * dataDim * sizeof(double));
    for (int i = 0; i < numClusters;i++)
    {
      int random = rand() % totalDataPoints;
      for (int j = 0; j < dataDim;j++)
      {
        centroids[i * dataDim + j] = data[random * dataDim + j];
      }
    }

    fclose(fp); // First I will try to read the file
    recv = (double*)malloc(dataPointsPerProcess * sizeof(double));
    sum = (double*)malloc(numClusters * dataDim * sizeof(double));
    totalcount = (int*)malloc(numClusters * sizeof(int));
    res_sum = (double*)malloc(numClusters * dataDim * sizeof(double));
    res_count = (int*)malloc(numClusters * sizeof(int));
    k_assignment = (int *)malloc(dataPointsPerProcess * sizeof(int));
    final_assignment = (int *)malloc(totalDataPoints * sizeof(int));
  }
  else
  {
     MPI_Bcast(&numClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&totalDataPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&dataDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
     dataPointsPerProcess = totalDataPoints / numClusters;
     recv = (double*)malloc(dataPointsPerProcess * sizeof(double));
     centroids = (double*)malloc(numClusters * dataDim * sizeof(double));
     sum = (double*)malloc(numClusters * dataDim * sizeof(double));
     totalcount = (int*)malloc(numClusters * sizeof(int));
     //res_sum = (double*)malloc(numClusters * dataDim * sizeof(double));
     //res_count = (int*)malloc(numClusters * sizeof(int));
     k_assignment = (int *)malloc(dataPointsPerProcess * sizeof(int));
  }
  MPI_Scatter(data, dataPointsPerProcess, MPI_DOUBLE, recv, dataPointsPerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int count = 0;
  while(count<iter)
  {
    MPI_Bcast(centroids, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // assign 0 to sum array
    for (int i = 0; i < numClusters; i++)
    {
      for (int j = 0; j < dataDim; j++)
      {
        sum[i * dataDim + j] = 0.0;
      }
      totalcount[i] = 0;
    }

    Calculatedist(recv, centroids, sum, dataDim,dataPointsPerProcess,numClusters,totalcount,k_assignment);
    // In the above function, I have sum for each centroid and now I have to perform element wise additiion
    MPI_Reduce(sum,res_sum,numClusters * dataDim,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(totalcount,res_count,numClusters,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Gather(k_assignment, dataPointsPerProcess, MPI_INT, final_assignment, dataPointsPerProcess, 0, MPI_COMM_WORLD);
    if(mpirank==0)
    {
      for (int k = 0; k < numClusters;k++)
      {
        for (int l = 0; l < dataDim;l++)
        {
          centroids[k * dataDim + l] = (double) res_sum[k * dataDim + l] / res_count[k];
        }
      }
      // calculate centroids
      // assign new centroids
    }
    count++;
  }
  if(mpirank==0)
  {
  for (k = 0; k < numClusters;k++)
  {
    for (int l = 0; l < dataDim;l++)
    {
      printf("%f",centroids[k * dataDim + l]);
      printf(" ");
    }
    printf("\n");
  }
  
  }
  free(recv);
  free(sum);
  free(totalcount);
  free(res_sum);
  free(res_count);
  if (mpirank == 0)
  {
    free(data);
    free(centroids);
  }

  MPI_Finalize();
  return 0;
}
