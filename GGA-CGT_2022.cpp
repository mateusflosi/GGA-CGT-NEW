/************************************************************************************************************************
GGA-CGT                                                                                  											*
GROUPING GENETIC ALGORITHM WITH CONTROLLED GENE TRANSMISSION FOR THE BIN PACKING PROBLEM 											*
************************************************************************************************************************/
/************************************************************************************************************************
 Author:	Marcela Quiroz-Castellanos																													*
			qc.marcela@gmail.com                                              															*
		 Tecnol�gico Nacional de M�xico                                    															*
		 Instituto Tecnol�gico de Ciudad Madero                            															*
		 Divisi�n de Estudios de Posgrado e Investigaci�n                  															*
		 Depto. de Sistemas y Computaci�n                                  															*
************************************************************************************************************************/
/************************************************************************************************************************
 The program excecutes GGA-CGT over a set instances using different configurations													*
 given by the user. Each configuration represents an independent execution of the GA.												*
																																								*
 Reference:                                                                               										*
		Quiroz-Castellanos, M., Cruz-Reyes, L., Torres-Jimenez, J.,																			*
		G�mez, C., Huacuja, H. J. F., & Alvim, A. C. (2015).																					*
	  A grouping genetic algorithm with controlled gene transmission for																*
	  the bin packing problem. Computers & Operations Research, 55, 52-64.																*
																																		*
 Input:                                                                                   										*
	File "instances.txt" including the name of the BPP instances to be solved; 		 												*
	Files including the standard instances to be solve;																						*
	File "configurations.txt" including the parameter values for each experiment;		 												*
																																	*
 Output:																																						*
	A set of files "GGA-CGT_(i).txt" including the experimental results for each														*
	configuration i in the input, stored in directory: Solutions_GGA-CGT;																*
   If(save_bestSolution = 1) a set of files HGGA_S_(i)_instance.txt including the													*
   obtained solution for each instance, for each configuration i, stored in directory: Details_GGA-CGT;						*
************************************************************************************************************************/
#include "linked_list.h"
#include <iostream>
#include <sstream>
#include <vector>

// CONSTANTS DEFINING THE SIZE OF THE PROBLEM
#define ATTRIBUTES 5000
#define P_size_MAX 500

char
	file[50],
	nameC[50];

int
	is_optimal_solution,
	save_bestSolution,
	repeated_fitness,
	max_gen,
	life_span,
	P_size,
	seed;

long int
	i,
	j,
	conf,
	number_items,
	generation,
	best_solution,
	n_,
	L2,
	bin_i,
	ordered_weight[ATTRIBUTES],
	permutation[ATTRIBUTES],
	items_auxiliary[ATTRIBUTES],
	ordered_population[P_size_MAX],
	best_individuals[P_size_MAX],
	random_individuals[P_size_MAX];

struct nodeData
{
	int index;
	int *conflitos;
	int conflitosSize;
	TIPO weight;
};

nodeData data[ATTRIBUTES];

unsigned long int
	higher_weight,
	lighter_weight,
	bin_capacity;

float
	p_m,
	p_c,
	k_ncs,
	k_cs,
	B_size,
	TotalTime;

long double
	total_accumulated_weight,
	weight1[ATTRIBUTES],
	_p_;

clock_t
	start,
	end;

FILE *output,
	*input_Configurations,
	*input_Instances;

struct SOLUTION
{
	linked_list L;
	double Bin_Fullness;
};

SOLUTION global_best_solution[ATTRIBUTES],
	population[P_size_MAX][ATTRIBUTES],
	children[P_size_MAX][ATTRIBUTES];

// Initial seeds for the random number generation
int seed_emptybin,
	seed_permutation;

// GA COMPONENTS
long int Generate_Initial_Population();
long int Generation();
void Gene_Level_Crossover_FFD(long int, long int, long int);
void Adaptive_Mutation_RP(long int, float, int);
void FF_n_(int);									 // First Fit with � pre-allocated-items (FF-�)
void RP(long int, long int &, long int[], long int); // Rearrangement by Pairs

// BPP Procedures
void FF(long int, SOLUTION[], long int &, long int, int);
void LowerBound();

// Auxiliary functions
void Find_Best_Solution();
void Sort_Ascending_IndividualsFitness();
void Sort_Descending_Weights(long int[], long int);
void Sort_Ascending_BinFullness(long int[], long int);
void Sort_Descending_BinFullness(long int[], long int);
void Sort_Random(long int[], long int, int);
void Copy_Solution(SOLUTION[], SOLUTION[], int);
void Clean_population();
long int Used_Items(long int, long int, long int[]);
void Adjust_Solution(long int);
long int LoadData();
void WriteOutput();
void sendtofile(SOLUTION[]);

// Pseudo-random number generator functions
int get_rand_ij(int *, int, int);
int get_rand(int *, int);
float randp(int *);
int trand();

int main()
{
	char aux[10], nombreC[30], string[50];
	system("mkdir Solutions_GGA-CGT");
	system("mkdir Details_GGA-CGT");

	// READING EACH CONFIGURATION IN FILE "configurations.txt", CONTAINING THE PARAMETER VALUES FOR EACH EXPERIMENT
	if ((input_Configurations = fopen("configurations.txt", "rt")) == NULL)
	{
		printf("\n INVALID FILE");
		// getch();
		exit(1);
	}
	fscanf(input_Configurations, "%[^\n]", string);
	while (!feof(input_Configurations))
	{
		fscanf(input_Configurations, "%ld", &conf);
		strcpy(nameC, "Solutions_GGA-CGT/GGA-CGT_(");
		// itoa(conf, aux, 10);
		strcat(nameC, aux);
		strcat(nameC, ").txt");
		output = fopen(nameC, "w+");
		fscanf(input_Configurations, "%d", &P_size);
		fscanf(input_Configurations, "%d", &max_gen);
		fscanf(input_Configurations, "%f", &p_m);
		fscanf(input_Configurations, "%f", &p_c);
		fscanf(input_Configurations, "%f", &k_ncs);
		fscanf(input_Configurations, "%f", &k_cs);
		fscanf(input_Configurations, "%f", &B_size);
		fscanf(input_Configurations, "%d", &life_span);
		fscanf(input_Configurations, "%d", &seed);
		fscanf(input_Configurations, "%d", &save_bestSolution);
		fprintf(output, "CONF\t|P|\tmax_gen\tn_m\tn_c\tk1(non-cloned_solutions)\tk2(cloned_solutions)\t|B|\tlife_span\tseed");
		fprintf(output, "\n%ld\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d", conf, P_size, max_gen, p_m, p_c, k_ncs, k_cs, B_size, life_span, seed);
		fprintf(output, "\nInstancias \t L2 \t Bins \t FBPP \t Gen \t Time");
		fclose(output);

		// READING FILE "instances.txt" CONTAINING THE NAME OF BPP INSTANCES TO PROCESS
		if ((input_Instances = fopen("instancesConflicts.txt", "rt")) == NULL)
		{
			printf("\n INVALID FILE");
			// getch();
			exit(1);
		}
		while (!feof(input_Instances))
		{
			fscanf(input_Instances, "%s", file);
			LoadData();
			for (i = 0; i < number_items; i++)
				ordered_weight[i] = i;
			Sort_Descending_Weights(ordered_weight, number_items);
			LowerBound();
			seed_permutation = seed;
			seed_emptybin = seed;
			for (i = 0; i < P_size; i++)
			{
				ordered_population[i] = i;
				random_individuals[i] = i;
				best_individuals[i] = i;
			}
			Clean_population();
			is_optimal_solution = 0;
			generation = 0;
			for (i = 0, j = n_; j < number_items; i++)
				permutation[i] = ordered_weight[j++];
			repeated_fitness = 0;
			// procedure GGA-CGT
			start = clock();
			if (!Generate_Initial_Population())
			{
				// Generate_Initial_Population() returns 1 if an optimal solution was found
				for (generation = 0; generation < max_gen; generation++)
				{
					if (Generation()) // Generation() returns 1 if an optimal solution was found
						break;
					Find_Best_Solution();
					// printf("\n %d", (int)global_best_solution[number_items + 1].Bin_Fullness);
				}
			}
			if (!is_optimal_solution) // is_optimal_solution is 1 if an optimal solution was printed before
			{
				end = clock();
				TotalTime = (end - start); // / (CLK_TCK * 1.0);
				Find_Best_Solution();
				WriteOutput();
			}
		}
		fclose(input_Instances);
	}
	fclose(input_Configurations);
	printf("\n\n\tEnd of process");
	getchar();

	return 0;
}

double GetFitness(SOLUTION solution[])
{
	return solution[number_items].Bin_Fullness;
}

void SetFitness(SOLUTION dest[], double value)
{
	dest[number_items].Bin_Fullness = value;
}

void SetFitness(SOLUTION dest[], SOLUTION origem[])
{
	SetFitness(dest, GetFitness(origem));
}

void IncrementFitness(SOLUTION solution[], long int individual)
{
	solution[number_items].Bin_Fullness += pow((solution[individual].Bin_Fullness / bin_capacity), 2);
}

double GetNumberOfBins(SOLUTION solution[])
{
	return solution[number_items + 1].Bin_Fullness;
}

void SetNumberOfBins(SOLUTION dest[], double value)
{
	dest[number_items + 1].Bin_Fullness = value;
}

void SetNumberOfBins(SOLUTION dest[], SOLUTION origem[])
{
	SetNumberOfBins(dest, GetNumberOfBins(origem));
}

double GetGeneration(SOLUTION solution[])
{
	return solution[number_items + 2].Bin_Fullness;
}

void SetGeneration(SOLUTION dest[], double value)
{
	dest[number_items + 2].Bin_Fullness = value;
}

void SetGeneration(SOLUTION dest[], SOLUTION origem[])
{
	SetGeneration(dest, GetGeneration(origem));
}

double GetNumberOfFullBins(SOLUTION solution[])
{
	return solution[number_items + 3].Bin_Fullness;
}

void AddNumberOfFullBins(SOLUTION dest[])
{
	dest[number_items + 3].Bin_Fullness++;
}

void SetNumberOfFullBins(SOLUTION dest[], double value)
{
	dest[number_items + 3].Bin_Fullness = value;
}

void SetNumberOfFullBins(SOLUTION dest[], SOLUTION origem[])
{
	SetNumberOfFullBins(dest, GetNumberOfFullBins(origem));
}

double GetHighestAvaliableCapacity(SOLUTION solution[])
{
	return solution[number_items + 4].Bin_Fullness;
}

void SetHighestAvaliableCapacity(SOLUTION dest[], double value)
{
	dest[number_items + 4].Bin_Fullness = value;
}

void SetHighestAvaliableCapacity(SOLUTION dest[], SOLUTION origem[])
{
	SetHighestAvaliableCapacity(dest, GetHighestAvaliableCapacity(origem));
}

long int GetWeight(nodeData *data)
{
	return data->weight;
}

int GetIndex(nodeData *data)
{
	return data->index;
}

int GetConflitosSize(nodeData *data)
{
	return data->conflitosSize;
}

int *GetConflitos(nodeData *data)
{
	return data->conflitos;
}

std::vector<int> GetConflitosVector(nodeData *data)
{
	int *ptr = GetConflitos(data);
	std::vector<int> vec(ptr, ptr + GetConflitosSize(data));
	return vec;
}

void SetIndex(nodeData *data, int value)
{
	data->index = value;
}

void SetWeight(nodeData *data, long int value)
{
	data->weight = value;
}

void SetConflitos(nodeData *data, int *value)
{
	data->conflitos = value;
}

void SetConflitosSize(nodeData *data, int value)
{
	data->conflitosSize = value;
}

long int CheckOptimalSolution(SOLUTION solution[], int add)
{
	SetGeneration(solution, generation + add);
	SetFitness(solution, GetFitness(solution) / GetNumberOfBins(solution));
	if (GetNumberOfBins(solution) == L2)
	{
		end = clock();
		Copy_Solution(global_best_solution, solution, 0);
		SetFitness(global_best_solution, solution);
		SetGeneration(global_best_solution, generation + add);
		SetNumberOfBins(global_best_solution, solution);
		SetNumberOfFullBins(global_best_solution, solution);
		TotalTime = (end - start); // / (CLK_TCK * 1.0);
		WriteOutput();
		is_optimal_solution = 1;
		return (1);
	}

	return (0);
}

/************************************************************************************************************************
 To generate an initial population P of individuals with FF-� packing heuristic.														*
  population[i][number_items].Bin_Fullness: Saves the fitness of the solution i                                        	*
  population[i][number_items + 1].Bin_Fullness:                                                                       	*
	Saves the total number of bins in the solution i                                                                    	*
  population[i][number_items + 2].Bin_Fullness:                                                                         *
	Saves the generation in which the solution i was generated                                                         	*
  population[i][number_items + 3].Bin_Fullness:                                                                        	*
	Saves the number of bins in the solution i that are fully at 100%                                                   	*
  population[i][number_items + 4].Bin_Fullness:																								  	*
	Saves the fullness of the bin with the highest avaliable capacity in the solution i                               	*
 Output:                                                                                        								*
	(1) when it finds a solution for which the size matches the L2 lower bound															*
   (0) otherwise																																			*
************************************************************************************************************************/
long int Generate_Initial_Population()
{
	for (i = 0; i < P_size; i++)
	{
		FF_n_(i);
		if (CheckOptimalSolution(population[i], 0))
			return (1);
	}
	return 0;
}

/************************************************************************************************************************
 To apply the reproduction technique: Controlled selection and Controlled replacement.          								*
 Output:                                                                                        								*
	(1) when it finds a solution for which the size matches the L2 lower bound     													*
   (2) if more than 0.1*P_size individuals (solutions) have duplicated-fitness														*
   (0) otherwise																																			*
************************************************************************************************************************/
long int Generation()
{
	long int
		f1,
		f2,
		h,
		k;

	/*-----------------------------------------------------------------------------------------------------
	---------------------------------Controlled selection for crossover------------------------------------
	-----------------------------------------------------------------------------------------------------*/
	Sort_Ascending_IndividualsFitness();
	// if(generation > 1 && repeated_fitness > 0.1*P_size)
	//	return (2);
	Sort_Random(random_individuals, 0, (int)(P_size - (int)(P_size * B_size)));
	Sort_Random(best_individuals, (1 - p_c) * P_size, P_size);
	k = 0;
	h = P_size - 1;
	for (i = P_size - 1, j = 0; i > P_size - (p_c / 2 * P_size); i--, j += 2)
	{
		f1 = best_individuals[h--];
		f2 = random_individuals[k++];
		if (f2 == f1)
		{
			f1 = best_individuals[h--];
		}
		Gene_Level_Crossover_FFD(ordered_population[f1], ordered_population[f2], j);
		if (CheckOptimalSolution(children[j], 1))
			return (1);
		Gene_Level_Crossover_FFD(ordered_population[f2], ordered_population[f1], j + 1);
		if (CheckOptimalSolution(children[j + 1], 1))
			return (1);
	}

	/*-----------------------------------------------------------------------------------------------------
	---------------------------------Controlled replacement for crossover----------------------------------
	-----------------------------------------------------------------------------------------------------*/
	k = 0;
	for (j = 0; j < p_c / 2 * P_size - 1; j++)
		Copy_Solution(population[ordered_population[random_individuals[k++]]], children[j], 1);
	k = 0;
	for (i = P_size - 1; i > P_size - (p_c / 2 * P_size); i--, j++)
	{
		while (GetGeneration(population[ordered_population[k]]) == generation + 1)
			k++;
		Copy_Solution(population[ordered_population[k++]], children[j], 1);
	}
	/*-----------------------------------------------------------------------------------------------------
   --------------------------------Controlled selection for mutation--------------------------------------
   -----------------------------------------------------------------------------------------------------*/
	Sort_Ascending_IndividualsFitness();
	// if(generation > 1 && repeated_fitness > 0.1*P_size)
	//	return (2);
	j = 0;
	for (i = P_size - 1; i > P_size - (p_m * P_size); i--)
	{
		if (i != j && j < (int)(P_size * B_size) && generation + 1 - GetGeneration(population[ordered_population[i]]) < life_span)
		{ /*-----------------------------------------------------------------------------------------------------
			  ----------------------------------Controlled replacement for mutation----------------------------------
		  -----------------------------------------------------------------------------------------------------*/
			Copy_Solution(population[ordered_population[j]], population[ordered_population[i]], 0);
			Adaptive_Mutation_RP(ordered_population[j], k_cs, 1);
			if (CheckOptimalSolution(population[ordered_population[j]], 1))
				return (1);
			j++;
		}
		else
		{
			Adaptive_Mutation_RP(ordered_population[i], k_ncs, 0);
			if (CheckOptimalSolution(population[ordered_population[i]], 1))
				return (1);
		}
	}
	return 0;
}

void CloneBinOfFather(long int child, long int *k2, long int father, long int random_order[], long int k, long int items[])
{
	long int ban = Used_Items(father, random_order[k], items);
	if (ban == 1)
	{
		children[child][*k2].L.clone_linked_list(population[father][random_order[k]].L);
		children[child][(*k2)++].Bin_Fullness = population[father][random_order[k]].Bin_Fullness;
		if (children[child][(*k2) - 1].Bin_Fullness < GetHighestAvaliableCapacity(children[child]))
			SetHighestAvaliableCapacity(children[child], children[child][(*k2) - 1].Bin_Fullness);
	}
}

/************************************************************************************************************************
 To recombine two parent solutions producing a child solution.          																*
 Input:                                                                                       									*
	The positions in the population of the two parent solutions: father_1 and father_2												*
	The position in the set of children of the child solution: child								   									*
************************************************************************************************************************/
void Gene_Level_Crossover_FFD(long int father_1, long int father_2, long int child)
{
	long int k,
		counter,
		k2 = 0,
		items[ATTRIBUTES] = {0},
		free_items[ATTRIBUTES] = {0};
	SetHighestAvaliableCapacity(children[child], bin_capacity);

	if (GetNumberOfBins(population[father_1]) > GetNumberOfBins(population[father_2]))
		counter = GetNumberOfBins(population[father_1]);
	else
		counter = GetNumberOfBins(population[father_2]);

	long int *random_order1 = new long int[counter];
	long int *random_order2 = new long int[counter];

	for (k = 0; k < counter; k++)
	{
		random_order1[k] = k;
		random_order2[k] = k;
	}

	Sort_Random(random_order1, 0, GetNumberOfBins(population[father_1]));
	Sort_Random(random_order2, 0, GetNumberOfBins(population[father_2]));
	Sort_Descending_BinFullness(random_order1, father_1);
	Sort_Descending_BinFullness(random_order2, father_2);

	for (k = 0; k < GetNumberOfBins(population[father_1]); k++)
	{
		if (population[father_1][random_order1[k]].Bin_Fullness >= population[father_2][random_order2[k]].Bin_Fullness)
		{
			CloneBinOfFather(child, &k2, father_1, random_order1, k, items);
			if (population[father_2][random_order2[k]].Bin_Fullness > 0)
			{
				CloneBinOfFather(child, &k2, father_2, random_order2, k, items);
			}
		}
		else
		{
			if (population[father_2][random_order2[k]].Bin_Fullness > 0)
			{
				CloneBinOfFather(child, &k2, father_2, random_order2, k, items);
			}
			CloneBinOfFather(child, &k2, father_1, random_order1, k, items);
		}
	}
	k = 0;
	for (counter = 0; counter < number_items; counter++)
	{
		if (items[ordered_weight[counter]] == 0)
			free_items[k++] = ordered_weight[counter];
	}
	if (k > 0)
	{
		bin_i = 0;
		for (counter = 0; counter < k - 1; counter++)
			FF(free_items[counter], children[child], k2, bin_i, 0);
		FF(free_items[counter], children[child], k2, bin_i, 1);
	}
	else
		for (k = 0; k < k2; k++)
			IncrementFitness(children[child], k);
	SetNumberOfBins(children[child], k2);
	free(random_order1);
	free(random_order2);
}

/************************************************************************************************************************
 To produce a small modification in a solution.						 																			*
 Input:                                                                                       									*
	The position in the population of the solution to mutate: individual 																*
	The rate of change to calculate the number of bins to eliminate: k	 																*
	A value that indicates if the solution was cloned: is_cloned																			*
************************************************************************************************************************/
void Adaptive_Mutation_RP(long int individual, float k, int is_cloned)
{
	long int
		number_bins,
		i,
		lightest_bin = 0,
		number_free_items = 0,
		free_items[ATTRIBUTES] = {0},
		ordered_BinFullness[ATTRIBUTES] = {0};
	node *p;
	for (i = 0; i < GetNumberOfBins(population[individual]); i++)
		ordered_BinFullness[i] = i;
	if (is_cloned)
		Sort_Random(ordered_BinFullness, 0, GetNumberOfBins(population[individual]));
	Sort_Ascending_BinFullness(ordered_BinFullness, individual);
	i = 1;
	while ((unsigned long int)population[individual][ordered_BinFullness[i]].Bin_Fullness < bin_capacity && i < GetNumberOfBins(population[individual]))
		i++;
	_p_ = 1 / (float)(k);
	number_bins = (long int)ceil(i * ((2 - i / GetNumberOfBins(population[individual])) / pow(i, _p_)) * (1 - ((double)get_rand(&seed_emptybin, (long int)ceil((1 / pow(i, _p_)) * 100)) / 100)));
	for (i = 0; i < number_bins; i++)
	{
		p = population[individual][ordered_BinFullness[lightest_bin]].L.first;
		while (p != NULL)
		{
			free_items[number_free_items++] = p->data;
			p = p->next;
		}
		population[individual][ordered_BinFullness[lightest_bin]].L.free_linked_list();
		population[individual][ordered_BinFullness[lightest_bin]].Bin_Fullness = 0;
		lightest_bin++;
	}

	SetNumberOfBins(population[individual], GetNumberOfBins(population[individual]) - number_bins);
	number_bins = GetNumberOfBins(population[individual]);
	Adjust_Solution(individual);
	RP(individual, number_bins, free_items, number_free_items);
	SetNumberOfBins(population[individual], number_bins);
}

/************************************************************************************************************************
 To generate a random BPP solution with the � large items packed in separate bins.													*
 Input:                                                                                       									*
	The position in the population of the new solution: individual 																		*
************************************************************************************************************************/
void FF_n_(int individual)
{
	long int
		i,
		j = 0,
		total_bins = 0;
	bin_i = 0;
	SetNumberOfFullBins(population[individual], 0.0);
	SetHighestAvaliableCapacity(population[individual], bin_capacity);
	if (n_ > 0)
	{
		for (i = 0; i < n_; i++)
		{
			population[individual][i].Bin_Fullness = GetWeight(&data[ordered_weight[i]]);
			population[individual][i].L.insert(ordered_weight[i]);
			total_bins++;
			if (population[individual][i].Bin_Fullness < GetHighestAvaliableCapacity(population[individual]))
				SetHighestAvaliableCapacity(population[individual], population[individual][i].Bin_Fullness);
		}
		i = number_items - i;
		Sort_Random(permutation, 0, i);
		for (j = 0; j < i - 1; j++)
			FF(permutation[j], population[individual], total_bins, bin_i, 0);
		FF(permutation[j], population[individual], total_bins, bin_i, 1);
	}
	else
	{
		Sort_Random(permutation, 0, number_items);
		for (j = 0; j < number_items - 1; j++)
			FF(permutation[j], population[individual], total_bins, bin_i, 0);
		FF(permutation[j], population[individual], total_bins, bin_i, 1);
	}
	SetNumberOfBins(population[individual], total_bins);
}

bool BinInConflito(std::vector<int> conflitos, int bin)
{
	for (int i = 0; i < conflitos.size(); ++i)
	{
		if (conflitos[i] == bin)
			return true;
	}
	return false;
}

bool BinInConflitoBuscaBinaria(int conflitosIndex, int binIndex)
{
	int *conflitos = data[conflitosIndex].conflitos;
	int bin = data[binIndex].index;
	int esquerda = 0;
	int direita = data[conflitosIndex].conflitosSize - 1;

	while (esquerda <= direita)
	{
		int meio = esquerda + (direita - esquerda) / 2;

		// Verifica se o valor está presente no meio
		if (conflitos[meio] == bin)
			return true;

		// Se o valor é maior que o valor do meio, ignora a metade esquerda
		if (conflitos[meio] < bin)
			esquerda = meio + 1;
		// Se o valor é menor que o valor do meio, ignora a metade direita
		else
			direita = meio - 1;
	}

	// Retorna -1 se o valor não for encontrado
	return false;
}

bool HaConflitoNoBin(std::vector<int> conflitos, std::vector<int> bins)
{
	for (int i = 0; i < bins.size(); ++i)
	{
		if (BinInConflito(conflitos, bins[i]))
			return true;
	}
	return false;
}

bool ValidChange(long int F[], long int k, node *ori, node *p, node *s, unsigned long int sum)
{
	long weightFk = data[F[k]].weight;
	long weightPS = data[p->data].weight + data[s->data].weight;
	if (sum - weightPS + weightFk > bin_capacity || weightFk < weightPS)
		return false;
	int pIndex = data[p->data].index;
	int sIndex = data[s->data].index;
	node *aux = ori;

	while (aux != NULL)
	{
		int auxIndex = data[aux->data].index;
		if (auxIndex != pIndex && auxIndex != sIndex)
		{
			if (BinInConflitoBuscaBinaria(aux->data, F[k]))
				return false;
			if (BinInConflitoBuscaBinaria(F[k], aux->data))
				return false;
		}
		aux = aux->next;
	}

	return true;
}

bool ValidDoubleChange(long int F[], long int k, long int k2, node *ori, node *p, node *s, unsigned long int sum)
{
	long weightFk = data[F[k]].weight;
	long weighFk2 = data[F[k2]].weight;
	long weightP = data[p->data].weight;
	long weightS = data[s->data].weight;
	// Verifica se os dois por fora somados são maiores ou iguais que dois itens do bin
	if (!((weightFk + weighFk2 > weightP + weightS) || ((weightFk + weighFk2 == weightP + weightS) && !(weightFk == weightP || weightFk == weightS))))
		return false;
	// Verificar se a troca não irá estourar a cabacidade do bin
	if (sum - (weightP + weightS) + (weightFk + weighFk2) > bin_capacity)
		return false;
	// Verifica se os dois itens não tem conflitos entre si
	if (BinInConflitoBuscaBinaria(F[k], F[k2]) || BinInConflitoBuscaBinaria(F[k2], F[k]))
		return false;

	node *aux = ori;
	int pIndex = data[p->data].index;
	int sIndex = data[s->data].index;

	while (aux != NULL)
	{
		int auxIndex = data[aux->data].index;
		if (auxIndex != pIndex && auxIndex != sIndex)
		{
			// Verifica se os dois itens de fora não conflitam com os itens do bin
			if (BinInConflitoBuscaBinaria(aux->data, F[k]) || BinInConflitoBuscaBinaria(aux->data, F[k2]))
				return false;
			if (BinInConflitoBuscaBinaria(F[k], aux->data) || BinInConflitoBuscaBinaria(F[k2], aux->data))
				return false;
		}
		aux = aux->next;
	}

	return true;
}

/************************************************************************************************************************
 To reinsert free items into an incomplete BPP solution.																						*
 Input:                                                                                                                	*
	The position in the population of the incomplete solution where the free items must be reinserted: individual			*
   The number of bins of the partial_solution: b																								*
   A set of free items to be reinserted into the partial_solution: F																		*
   The number of free items of F: number_free_items																							*
************************************************************************************************************************/
void RP(long int individual, long int &b, long int F[], long int number_free_items)
{
	long int
		i,
		k,
		k2,
		ban,
		total_free = 0,
		ordered_BinFullness[ATTRIBUTES] = {0},
		*new_free_items = new long int[2];

	unsigned long int
		sum = 0;

	node *ori,
		*p,
		*s,
		*aux;

	higher_weight = GetWeight(&data[F[0]]);
	lighter_weight = GetWeight(&data[F[0]]);
	bin_i = b;
	SetFitness(population[individual], 0.0);
	SetNumberOfFullBins(population[individual], 0.0);
	SetHighestAvaliableCapacity(population[individual], bin_capacity);

	for (i = 0; i < b; i++)
		ordered_BinFullness[i] = i;
	Sort_Random(ordered_BinFullness, 0, b);
	Sort_Random(F, 0, number_free_items);

	for (i = 0; i < b; i++)
	{
		sum = (long int)population[individual][ordered_BinFullness[i]].Bin_Fullness;
		ori = population[individual][ordered_BinFullness[i]].L.first;
		p = population[individual][ordered_BinFullness[i]].L.first;
		while (p->next != NULL)
		{
			ban = 0;
			aux = p;
			s = p->next;
			while (s != NULL)
			{
				for (k = 0; k < number_free_items - 1; k++)
				{
					if (i == b - 1)
						if (GetWeight(&data[F[k]]) > higher_weight)
							higher_weight = GetWeight(&data[F[k]]);
					for (k2 = k + 1; k2 < number_free_items; k2++)
					{
						if (ValidChange(F, k, ori, p, s, sum))
						{
							sum = sum - (GetWeight(&data[p->data]) + GetWeight(&data[s->data])) + (GetWeight(&data[F[k]]));
							new_free_items[0] = p->data;
							new_free_items[1] = s->data;
							p->data = F[k];
							aux->next = s->next;
							free(s);
							if (population[individual][ordered_BinFullness[i]].L.last == s)
								population[individual][ordered_BinFullness[i]].L.last = aux;
							population[individual][ordered_BinFullness[i]].L.num--;
							F[k] = new_free_items[0];
							F[number_free_items + total_free] = new_free_items[1];
							total_free++;
							ban = 1;
							break;
						}
						if (ValidChange(F, k2, ori, p, s, sum))
						{
							sum = sum - (GetWeight(&data[p->data]) + GetWeight(&data[s->data])) + (GetWeight(&data[F[k2]]));
							new_free_items[0] = p->data;
							new_free_items[1] = s->data;
							p->data = F[k2];
							aux->next = s->next;
							free(s);
							if (population[individual][ordered_BinFullness[i]].L.last == s)
								population[individual][ordered_BinFullness[i]].L.last = aux;
							population[individual][ordered_BinFullness[i]].L.num--;
							F[k2] = new_free_items[0];
							F[number_free_items + total_free] = new_free_items[1];
							total_free++;
							ban = 1;
							break;
						}
						if (ValidDoubleChange(F, k, k2, ori, p, s, sum))
						{
							sum = sum - (GetWeight(&data[p->data]) + GetWeight(&data[s->data])) + (GetWeight(&data[F[k]]) + GetWeight(&data[F[k2]]));
							new_free_items[0] = p->data;
							new_free_items[1] = s->data;
							p->data = F[k];
							s->data = F[k2];
							F[k] = new_free_items[0];
							F[k2] = new_free_items[1];
							if (sum == bin_capacity)
							{
								ban = 1;
								break;
							}
						}
					}
					if (ban)
						break;
				}
				if (ban)
					break;
				aux = s;
				s = s->next;
			}
			if (ban)
				break;
			p = p->next;
		}
		population[individual][ordered_BinFullness[i]].Bin_Fullness = sum;
		if (population[individual][ordered_BinFullness[i]].Bin_Fullness < GetHighestAvaliableCapacity(population[individual]))
			SetHighestAvaliableCapacity(population[individual], population[individual][ordered_BinFullness[i]].Bin_Fullness);
		if ((unsigned long int)population[individual][ordered_BinFullness[i]].Bin_Fullness == bin_capacity)
			AddNumberOfFullBins(population[individual]);
		else if ((unsigned long int)population[individual][ordered_BinFullness[i]].Bin_Fullness + GetWeight(&data[ordered_weight[number_items - 1]]) <= bin_capacity)
		{
			if (ordered_BinFullness[i] < bin_i)
				bin_i = ordered_BinFullness[i];
		}
	}
	for (i = 0; i < bin_i; i++)
		IncrementFitness(population[individual], i);

	free(new_free_items);
	number_free_items += total_free;

	if (higher_weight < .5 * bin_capacity)
		Sort_Random(F, 0, number_free_items);
	else
	{
		Sort_Descending_Weights(F, number_free_items);
		lighter_weight = GetWeight(&data[F[number_free_items - 1]]);
	}

	if (lighter_weight > bin_capacity - (unsigned long int)GetHighestAvaliableCapacity(population[individual]))
	{
		for (i = bin_i; i < b; i++)
			IncrementFitness(population[individual], i);
		bin_i = b;
	}
	for (i = 0; i < number_free_items - 1; i++)
		FF(F[i], population[individual], b, bin_i, 0);
	FF(F[i], population[individual], b, bin_i, 1);
}

bool ValidInsert(SOLUTION individual, long int item)
{
	if ((unsigned long int)individual.Bin_Fullness + data[item].weight > bin_capacity)
		return false;

	node *aux = individual.L.first;
	while (aux != NULL)
	{
		if (BinInConflitoBuscaBinaria(aux->data, item))
			return false;
		if (BinInConflitoBuscaBinaria(item, aux->data))
			return false;
		aux = aux->next;
	}

	return true;
}

/************************************************************************************************************************
 To insert an item into an incomplete BPP solution.																							*
 Input:                                                                                       									*
   An item to be inserted into the individual: item																							*
	An incomplete chromosome where the item must be inserted: individual					  			  									*
   The number of bins of the individual: total_bins																							*
   The first bin that could have sufficient available capacity to store the item: beginning										*
   A value that indicates if it is the last item to be stored into the individual: is_last										*
************************************************************************************************************************/
void FF(long int item, SOLUTION individual[], long int &total_bins, long int beginning, int is_last)
{
	long int i;

	if (!is_last && GetWeight(&data[item]) > (bin_capacity - (unsigned long int)GetHighestAvaliableCapacity(individual)))
		i = total_bins;
	else
		for (i = beginning; i < total_bins; i++)
		{
			if (ValidInsert(individual[i], item))
			{
				individual[i].Bin_Fullness += GetWeight(&data[item]);
				individual[i].L.insert(item);
				if ((unsigned long int)individual[i].Bin_Fullness == bin_capacity)
					AddNumberOfFullBins(individual);
				if (is_last)
				{
					for (i; i < total_bins; i++)
						IncrementFitness(individual, i);
					return;
				}
				if ((unsigned long int)individual[i].Bin_Fullness + GetWeight(&data[ordered_weight[number_items - 1]]) > bin_capacity && i == bin_i)
				{
					bin_i++;
					IncrementFitness(individual, i);
				}
				return;
			}
			if (is_last)
				IncrementFitness(individual, i);
		}
	individual[i].Bin_Fullness += GetWeight(&data[item]);
	individual[i].L.insert(item);
	if (individual[i].Bin_Fullness < GetHighestAvaliableCapacity(individual))
		SetHighestAvaliableCapacity(individual, individual[i].Bin_Fullness);
	if (is_last)
		IncrementFitness(individual, i);
	total_bins++;
}

/************************************************************************************************************************
 To calculate the lower bound L2 of Martello and Toth and the � large items n_														*
************************************************************************************************************************/
void LowerBound()
{
	long int k, m, i, j, aux1, aux2;
	long double sjx = 0, sj2 = 0, sj3 = 0;
	long int jx = 0, cj12, jp = 0, jpp = 0, cj2;

	while (GetWeight(&data[ordered_weight[jx]]) > bin_capacity / 2 && jx < number_items)
		jx++;
	n_ = jx;
	if (jx == number_items)
	{
		L2 = jx;
		return;
	}
	if (jx == 0)
	{
		if (fmod(total_accumulated_weight, bin_capacity) >= 1)
			L2 = (long int)ceil(total_accumulated_weight / bin_capacity);
		else
			L2 = (long int)(total_accumulated_weight / bin_capacity);
		return;
	}
	else
	{
		cj12 = jx;
		for (i = jx; i < number_items; i++)
			sjx += GetWeight(&data[ordered_weight[i]]);
		jp = jx;
		for (i = 0; i < jx; i++)
		{
			if (GetWeight(&data[ordered_weight[i]]) <= bin_capacity - GetWeight(&data[ordered_weight[jx]]))
			{
				jp = i;
				break;
			}
		}

		cj2 = jx - jp;
		for (i = jp; i <= jx - 1; i++)
			sj2 += GetWeight(&data[ordered_weight[i]]);
		jpp = jx;
		sj3 = GetWeight(&data[ordered_weight[jpp]]);
		ordered_weight[number_items] = number_items;
		SetWeight(&data[number_items], 0);
		while (GetWeight(&data[ordered_weight[jpp + 1]]) == GetWeight(&data[ordered_weight[jpp]]))
		{
			jpp++;
			sj3 += GetWeight(&data[ordered_weight[jpp]]);
		}
		L2 = cj12;

		do
		{
			if (fmod((sj3 + sj2), bin_capacity) >= 1)
				aux1 = (long int)ceil((sj3 + sj2) / bin_capacity - cj2);
			else
				aux1 = (long int)((sj3 + sj2) / bin_capacity - cj2);

			if (L2 < (cj12 + aux1))
				L2 = cj12 + aux1;
			jpp++;
			if (jpp < number_items)
			{
				sj3 += GetWeight(&data[ordered_weight[jpp]]);
				while (GetWeight(&data[ordered_weight[jpp + 1]]) == GetWeight(&data[ordered_weight[jpp]]))
				{
					jpp++;
					sj3 += GetWeight(&data[ordered_weight[jpp]]);
				}
				while (jp > 0 && GetWeight(&data[ordered_weight[jp - 1]]) <= bin_capacity - GetWeight(&data[ordered_weight[jpp]]))
				{
					jp--;
					cj2++;
					sj2 += GetWeight(&data[ordered_weight[jp]]);
				}
			}
			if (fmod((sjx + sj2), bin_capacity) >= 1)
				aux2 = (long int)ceil((sjx + sj2) / bin_capacity - cj2);
			else
				aux2 = (long int)((sjx + sj2) / bin_capacity - cj2);
		} while (jpp <= number_items || (cj12 + aux2) > L2);
	}
}

/************************************************************************************************************************
 To find the solution with the highest fitness of the population and update the global_best_solution							*
************************************************************************************************************************/
void Find_Best_Solution()
{
	long int i,
		best_individual = 0;
	for (i = 0; i < P_size; i++)
	{
		if (GetFitness(population[i]) > GetFitness(population[best_individual]))
			best_individual = i;
	}
	if (generation + 1 > 1)
	{
		if (GetFitness(population[best_individual]) > GetFitness(global_best_solution))
		{
			Copy_Solution(global_best_solution, population[best_individual], 0);
		}
	}
	else
	{
		Copy_Solution(global_best_solution, population[best_individual], 0);
	}
}

/************************************************************************************************************************
 To sort the individuals of the population in ascending order of their fitness													  	*
************************************************************************************************************************/
void Sort_Ascending_IndividualsFitness()
{
	long int i,
		k = P_size - 1,
		i2 = 0,
		aux,
		ban = 1;

	while (ban)
	{
		ban = 0;
		for (i = i2; i < k; i++)
		{
			if (GetFitness(population[ordered_population[i]]) > GetFitness(population[ordered_population[i + 1]]))
			{
				aux = ordered_population[i];
				ordered_population[i] = ordered_population[i + 1];
				ordered_population[i + 1] = aux;
				ban = 1;
			}
			else if (GetFitness(population[ordered_population[i]]) == GetFitness(population[ordered_population[i + 1]]))
			{
				aux = ordered_population[i + 1];
				ordered_population[i + 1] = ordered_population[i2];
				ordered_population[i2] = aux;
				i2++;
			}
		}
		k--;
	}
	repeated_fitness = i2;
}

/************************************************************************************************************************
 To sort the bins of a solution in ascending order of their filling																		*
 Input:                                                                                                                	*
	An array to save the order of the bins: ordered_BinFullness																				*                                                                                                                	*
	The position in the population of the solution: individual																				*
************************************************************************************************************************/
void Sort_Ascending_BinFullness(long int ordered_BinFullness[], long int individual)
{
	long int m,
		k,
		temporary_variable,
		ban = 1;

	k = GetNumberOfBins(population[individual]) - 1;
	while (ban)
	{
		ban = 0;
		for (m = 0; m < k; m++)
		{
			if (population[individual][ordered_BinFullness[m]].Bin_Fullness > population[individual][ordered_BinFullness[m + 1]].Bin_Fullness)
			{
				temporary_variable = ordered_BinFullness[m];
				ordered_BinFullness[m] = ordered_BinFullness[m + 1];
				ordered_BinFullness[m + 1] = temporary_variable;
				ban = 1;
			}
		}
		k--;
	}
}

/************************************************************************************************************************
 To sort the bins of a solution in descending order of their filling																		*
 Input:                                                                                                                	*
	An array to save the order of the bins: ordered_BinFullness																				*                                                                                                                	*
	The position in the population of the solution: individual																				*
************************************************************************************************************************/
void Sort_Descending_BinFullness(long int ordered_BinFullness[], long int individual)
{
	long int m,
		k,
		temporary_variable,
		ban = 1;

	k = GetNumberOfBins(population[individual]) - 1;
	while (ban)
	{
		ban = 0;
		for (m = 0; m < k; m++)
		{
			if (population[individual][ordered_BinFullness[m]].Bin_Fullness < population[individual][ordered_BinFullness[m + 1]].Bin_Fullness)
			{
				temporary_variable = ordered_BinFullness[m];
				ordered_BinFullness[m] = ordered_BinFullness[m + 1];
				ordered_BinFullness[m + 1] = temporary_variable;
				ban = 1;
			}
		}
		k--;
	}
}

/************************************************************************************************************************
 To sort the elements between the positions [k] and [n] of an array in random order													*
 Input:                                                                                                                	*
	The array to be randomized: random_array																										*                                                                                                                	*
	The initial random position: k																													*
	The final random position: n																														*
************************************************************************************************************************/
void Sort_Random(long int random_array[], long int k, int n)
{
	long int i,
		aux,
		random_number;

	for (i = n - 1; i >= k; i--)
	{
		random_number = k + get_rand(&seed_permutation, n - k) - 1;
		aux = random_array[random_number];
		random_array[random_number] = random_array[i];
		random_array[i] = aux;
		if (GetWeight(&data[random_array[i]]) < lighter_weight)
			lighter_weight = GetWeight(&data[random_array[i]]);
		if (GetWeight(&data[random_array[random_number]]) < lighter_weight)
			lighter_weight = GetWeight(&data[random_array[random_number]]);
	}
}

/************************************************************************************************************************
 To sort a set of items in descending order of their weights																				*
 Input:                                                                                                                	*
	An array to save the order of the items: ordered_weight																					*                                                                                                                	*
	The number of items in the set: n																												*
************************************************************************************************************************/
void Sort_Descending_Weights(long int ordered_weight[], long int n)
{
	long int m,
		k,
		temporary_variable,
		ban = 1;

	k = n - 1;
	while (ban)
	{
		ban = 0;
		for (m = 0; m < k; m++)
		{
			if (GetWeight(&data[ordered_weight[m]]) < GetWeight(&data[ordered_weight[m + 1]]))
			{
				temporary_variable = ordered_weight[m];
				ordered_weight[m] = ordered_weight[m + 1];
				ordered_weight[m + 1] = temporary_variable;
				ban = 1;
			}
		}
		k--;
	}
}

/************************************************************************************************************************
 To copy solution2 in solution																														*
 Input:                                                                                                                	*
	A solution to save the copied solution: solution																							*                                                                                                                	*
	The solution to be copied: solution2																										  	*
	A value that indicates if the copied solution must be deleted: delete_solution2													*
************************************************************************************************************************/
void Copy_Solution(SOLUTION solution[], SOLUTION solution2[], int delete_solution2)
{
	long int j;

	for (j = 0; j < GetNumberOfBins(solution2); j++)
	{
		solution[j].Bin_Fullness = solution2[j].Bin_Fullness;
		solution[j].L.clone_linked_list(solution2[j].L);
		if (delete_solution2)
		{
			solution2[j].Bin_Fullness = 0;
			solution2[j].L.free_linked_list();
		}
	}
	while (j < GetNumberOfBins(solution))
	{
		solution[j].Bin_Fullness = 0;
		solution[j++].L.free_linked_list();
	}
	SetFitness(solution, solution2);
	SetNumberOfBins(solution, solution2);
	SetGeneration(solution, solution2);
	SetNumberOfFullBins(solution, solution2);
	SetHighestAvaliableCapacity(solution, solution2);
	if (delete_solution2)
	{
		SetFitness(solution2, 0.0);
		SetNumberOfBins(solution2, 0.0);
		SetGeneration(solution2, 0.0);
		SetNumberOfBins(solution2, 0.0);
		SetHighestAvaliableCapacity(solution2, 0.0);
	}
}

/************************************************************************************************************************
 To free the memory of the individuals of the population																						*
************************************************************************************************************************/
void Clean_population()
{
	long int i,
		j;

	for (i = 0; i < P_size; i++)
	{
		for (j = 0; j < number_items + 5; j++)
		{
			population[i][j].L.free_linked_list();
			population[i][j].Bin_Fullness = 0;
			children[i][j].L.free_linked_list();
			children[i][j].Bin_Fullness = 0;
		}
	}
}

/************************************************************************************************************************
 To check if any of the items of the current bin is already in the solution															*
 Input:                                                                                                                	*
	The position in the population of the solution: individual																				*
	A new bin that could be added to the solution: bin																							*
   An array that indicates the items that are already in the solution: items                                            *
 Output:                                                                                                                *
	(1) when none of the items in the current bin is already in the solution     														*
   (0) otherwise																																			*
************************************************************************************************************************/
long int Used_Items(long int individual, long int bin, long int items[])
{
	long int item,
		i,
		counter = 0;
	node *p;

	p = population[individual][bin].L.first;
	while (p != NULL)
	{
		item = p->data;
		p = p->next;
		if (items[item] != 1)
		{
			items_auxiliary[counter++] = item;
			items[item] = 1;
		}
		else
		{
			for (i = 0; i < counter; i++)
				items[items_auxiliary[i]] = 0;
			return 0;
		}
	}
	return (1);
}

/************************************************************************************************************************
 To put together all the used bins of the solution																								*
 Input:                                                                                                                	*
	The position in the population of the solution: individual																				*
************************************************************************************************************************/
void Adjust_Solution(long int individual)
{
	long int i = 0,
			 j = 0,
			 k;
	while (population[individual][i].Bin_Fullness > 0)
		i++;
	for (j = i, k = i; j < number_items; j++, k++)
	{
		if (j < GetNumberOfBins(population[individual]))
		{
			while (population[individual][k].Bin_Fullness == 0)
				k++;
			population[individual][j].L.first = NULL;
			population[individual][j].L.last = NULL;
			population[individual][j].Bin_Fullness = population[individual][k].Bin_Fullness;
			population[individual][j].L.get_linked_list(population[individual][k].L);
		}
		else
		{
			population[individual][j].Bin_Fullness = 0;
			population[individual][j].L.first = NULL;
			population[individual][j].L.last = NULL;
			population[individual][j].L.num = 0;
		}
	}
}

std::vector<int> splitLine(char line[])
{
	std::vector<int> numbers;
	std::stringstream ss(line);
	int num;

	while (ss >> num)
	{
		numbers.push_back(num);
	}
	return numbers;
}

int *getConflitos(std::vector<int> numbers)
{
	int *array = (int *)malloc((numbers.size() - 2) * sizeof(int));

	for (i = 2; i < numbers.size(); i++)
	{
		array[i - 2] = numbers[i];
	}

	return array;
}

/************************************************************************************************************************
 To read the data defining a BPP instance																											*
************************************************************************************************************************/
long int LoadData()
{
	char string[300];
	char line[5000];
	long k;
	long int ban = 0;
	long double bin_capacity1;
	long double total_accumulated_aux = 0;

	FILE *data_file;

	string[0] = '\0';
	strcpy(string, file);
	if ((data_file = fopen(string, "rt")) == NULL)
	{
		printf("\nThere is no data file ==> [%s]%c", string, 7);
		return 0;
	}
	printf("\nThe file is %s\n", string);
	fgets(string, 300, data_file);
	fscanf(data_file, "%ld\n", &number_items);
	bin_capacity = 0;
	fgets(string, 300, data_file);
	fscanf(data_file, "%Lf\n", &bin_capacity1);
	best_solution = 0;
	fgets(string, 300, data_file);
	fscanf(data_file, "%ld\n", &best_solution);
	fgets(string, 300, data_file);
	total_accumulated_weight = 0;
	for (k = 0; k < number_items; k++)
	{
		fscanf(data_file, "%5000[^\n]\n", &line);
		std::vector<int> array = splitLine(line);
		weight1[k] = static_cast<long double>(array[1]);
		SetIndex(&data[k], array[0]);
		SetWeight(&data[k], (long int)weight1[k]);
		SetConflitos(&data[k], getConflitos(array));
		SetConflitosSize(&data[k], array.size() - 2);
		total_accumulated_weight = (total_accumulated_weight + GetWeight(&data[k]));
		total_accumulated_aux += weight1[k];
		if (ban == 0)
		{
			if (weight1[k] / GetWeight(&data[k]) > 1)
			{
				ban = 1;
			}
		}
	}
	if (ban)
	{
		total_accumulated_weight = 0;
		for (k = 0; k < number_items; k++)
		{
			SetWeight(&data[k], (long int)(ceil(weight1[k] * bin_capacity1 - .5)));
			total_accumulated_weight = (total_accumulated_weight + GetWeight(&data[k]));
		}
		bin_capacity1 *= bin_capacity1;
	}
	bin_capacity = (long int)bin_capacity1;
	fclose(data_file);
	if (ban)
	{
		if ((long int)total_accumulated_weight != (long int)(ceil(total_accumulated_aux * sqrt(bin_capacity) - .5)))
		{
			// getch();
			exit(1);
		}
	}
	return 1;
}

/************************************************************************************************************************
 To print the performance of the procedure on a BPP instance in a data file															*
************************************************************************************************************************/
void WriteOutput()
{
	output = fopen(nameC, "a");
	fprintf(output, "\n%s \t %d \t %d \t %f \t %ld \t %f", file, (int)L2, (int)GetNumberOfBins(global_best_solution), GetFitness(global_best_solution), generation, TotalTime);
	if (save_bestSolution == 1)
		sendtofile(global_best_solution);
	fclose(output);
}

char *getString1Clean(char *string1)
{
	int aux = -1;
	for (i = 0; i < 50; i++)
	{
		if (string1[i] == '\0')
			return &string1[aux + 1];
		if (string1[i] == '/')
			aux = i;
	}
	return string1;
}

/************************************************************************************************************************
 To print the global best solution in a data file																								*
************************************************************************************************************************/
void sendtofile(SOLUTION best[])
{
	char string1[50],
		fil[50],
		aux[10];

	long double accumulated = 0;
	long int bin,
		ban = 1,
		item = 0,
		position = 0;
	std::vector<std::vector<int>> conflitos;
	std::vector<std::vector<int>> indexes;
	unsigned long int
		bins[ATTRIBUTES] = {0},
		n_bins = GetNumberOfBins(best);

	int binError = -1,
		banError = 0;
	long int j;

	FILE *output;
	node *p;
	strcpy(fil, "Details_GGA-CGT/GGA-CGT_S_(");
	strcpy(string1, file);
	// itoa(conf, aux, 10);
	strcat(fil, aux);
	strcat(fil, ")_");
	strcat(fil, getString1Clean(string1));
	if ((output = fopen(fil, "w+")) == NULL)
	{
		// printf("\nThere is no data file ==> [%s]%c", file, 7);
		getchar();
		exit(1);
	}
	fprintf(output, "Instance:\t%s\n", file);
	fprintf(output, "Number of items:\t%ld\n", number_items);
	fprintf(output, "Bin capacity:\t%ld\n", bin_capacity);
	fprintf(output, "L2:\t%ld\n", L2);
	fprintf(output, "\n****************************GGA-CGT global best solution******************************\n");
	fprintf(output, "Number of bins:\n%ld\n", n_bins);
	fprintf(output, "Fitness:\n%f\n", best[number_items].Bin_Fullness);
	fprintf(output, "Optimal order of the weights:\n");
	for (bin = 0; bin < n_bins; bin++)
	{
		std::vector<int> conflitosBin;
		std::vector<int> indexesBin;
		bins[bin] = 0;
		p = best[bin].L.first;
		while (true)
		{
			if (p == NULL)
			{
				conflitos.push_back(conflitosBin);
				indexes.push_back(indexesBin);
				break;
			}
			item = p->data;
			p = p->next;

			bins[bin] += GetWeight(&data[item]);
			accumulated += GetWeight(&data[item]);
			fprintf(output, "%ld\n", GetWeight(&data[item]));
			if (bins[bin] > bin_capacity)
			{
				printf("ERROR the capacity of bin %ld was exceeded", bin);
				binError = bin;
				getchar();
				banError = 1;
			}

			if (BinInConflito(conflitosBin, GetIndex(&data[item])) || HaConflitoNoBin(GetConflitosVector(&data[item]), indexesBin))
			{
				printf("ERROR there is a conflict in the bin %ld", bin);
				binError = bin;
				getchar();
				banError = 1;
			}

			indexesBin.push_back(GetIndex(&data[item]));
			std::vector<int> conflitosItem = GetConflitosVector(&data[item]);
			conflitosBin.insert(conflitosBin.end(), conflitosItem.begin(), conflitosItem.end());
		}
	}

	if (accumulated != total_accumulated_weight)
	{
		printf("ERROR inconsistent sum of weights");
		getchar();
	}
	fprintf(output, "\nDetailed solution:");
	for (j = 0; j < n_bins; j++)
	{
		if (bins[j] > bin_capacity)
			fprintf(output, " \n ********************ERROR the capacity of the bin was exceeded******************");
		if (HaConflitoNoBin(conflitos[j], indexes[j]))
			fprintf(output, " \n ***********************ERROR there is a conflict in the bin*********************");

		fprintf(output, "\n\nBIN %ld\nFullness: %ld Gap: %ld\nStored items:\t ", j + 1, bins[j], bin_capacity - bins[j]);
		p = best[j].L.first;
		for (position = 0;; position++)
		{
			if (p == NULL)
				break;
			item = p->data;
			p = p->next;
			fprintf(output, "[Item: %ld, Weight: %ld]\t", item + 1, GetWeight(&data[item]));
		}
	}

	fclose(output);

	if (banError)
		exit(1);
}

/******************************************************
 Author: Adriana Cesario de Faria Alvim               *
		 (alvim@inf.puc-rio.br, adriana@pep.ufrj.br ) *
*******************************************************/
/***************************************************************************
 Portable pseudo-random number generator                                   *
 Machine independent as long as the machine can represent all the integers *
		 in the interval [- 2**31 + 1, 2**31 - 1].                         *
																		   *
 Reference: L. Schrage, "A more Portable Fortran Random Number Generator", *
			ACM Transactions on Mathematical Software, Vol. 2, No. 2,      *
			(June, 1979).                                                  *
																		   *
 The generator produces a sequence of positive integers, "ix",             *
	  by the recursion: ix(i+1) = A * ix(i) mod P, where "P" is Mersenne   *
	  prime number (2**31)-1 = 2147483647 and "A" = 7**5 = 16807. Thus all *
	  integers "ix" produced will satisfy ( 0 < ix < 2147483647 ).         *
																		   *
 The generator is full cycle, every integer from 1 to (2**31)-2 =          *
	  2147483646 is generated exactly once in the cycle.                   *
																		   *
 Input: integer "ix", ( 0 < ix < 2147483647 )                              *
 Ouput: real "xrand", ( 0 < xrand < 1 )                                    *
***************************************************************************/
float randp(int *ix)
{
	int xhi, xalo, leftlo, fhi, k;

	const int A = 16807;	  /* = 7**5		       */
	const int P = 2147483647; /* = Mersenne prime (2**31)-1  */
	const int b15 = 32768;	  /* = 2**15	               */
	const int b16 = 65536;	  /* = 2**16	               */

	/* get 15 hi order bits of ix */
	xhi = *ix / b16;

	/* get 16 lo bits of ix and form lo product */
	xalo = (*ix - xhi * b16) * A;

	/* get 15 hi order bits of lo product	*/
	leftlo = xalo / b16;

	/* from the 31 highest bits of full product */
	fhi = xhi * A + leftlo;

	/* get overflo past 31st bit of full product */
	k = fhi / b15;

	/* assemble all the parts and presubtract P */
	/* the parentheses are essential            */
	*ix = (((xalo - leftlo * b16) - P) + (fhi - k * b15) * b16) + k;

	/* add P back in if necessary  */
	if (*ix < 0)
		*ix = *ix + P;

	/* multiply by 1/(2**31-1) */
	return (float)(*ix * 4.656612875e-10);
}

/***********************************************************
 To generate a random integer "k" in [i,j].                *
 Input: integers "seed", "i" and "j" in [1,2147483646]     *
 Ouput: integer in [i,j]                                   *
************************************************************/
int get_rand_ij(int *seed, int i, int j)
{

	randp(seed);

	return ((double)*seed / ((double)2147483647 / ((double)(j - i + 1)))) + i;
}

/**************************************************************
 To generate a random integer "k" in [1,size].                *
 Input: integers "seed" and "size" in [1,2147483646]          *
 Ouput: integer in [1,size]
								   *
**************************************************************/
int get_rand(int *seed, int size)
{
	randp(seed);

	return ((double)*seed / ((double)2147483647 / ((double)size))) + 1;
}

/*************************************************
 To check the correctness of the implementation. *
 Output: correct (1) or wrong (0)                 *
**************************************************/
int trand()
{
	int i, seed;

	seed = 1;

	for (i = 0; i < 1000; i++)
		randp(&seed);

	if (seed == 522329230)
		return 1;
	else
		return 0;
}
