# include <population.h>	
# include <math.h>
# include <iostream>

//# define MAX_RULE	65536
# define MAX_RULE	256


/* Population constructor */
/* Input: genome count , genome size, pointer to Program instance */
Population::Population(int gcount,int gsize,Program *p)
{
	elitism=1;
	selection_rate = 0.1;
	mutation_rate  = 0.05;
	genome_count   = gcount;
	genome_size    = gsize;
	generation     = 0;
	program        = p;

	/* Create the population and based on genome count and size */
	/* Initialize the genomes to random */
	genome=new int*[genome_count];
	children=new int*[genome_count];
	for(int i=0;i<genome_count;i++)
	{
		genome[i]=new int[genome_size];
		children[i]=new int[genome_size];
			for(int j=0;j<genome_size;j++)
				genome[i][j]=rand()%MAX_RULE;
	}
	fitness_array=new double[genome_count];
}

/* Reinitialize the population to random */
void	Population::reset()
{
	generation = 0;
	for(int i=0;i<genome_count;i++)
		for(int j=0;j<genome_size;j++)
				genome[i][j]=rand()%MAX_RULE;
	for(int i=0;i<genome_count;i++)
			fitness_array[i]=-1e+8;
}

/* Return the fitness of a genome */
double 	Population::fitness(vector<int> &g)
{

	double f=program->fitness(g);
	return f;
}

/* The selection of the chromosomes according to the fitness value is performed */
void	Population::select()
{
	int itemp[genome_size];
	for(int i=0;i<genome_count;i++)
	{
		for(int j=0;j<genome_count-1;j++)
		{
			if(fitness_array[j+1]>fitness_array[j])
			{
				double dtemp;
				dtemp=fitness_array[j];
				fitness_array[j]=fitness_array[j+1];
				fitness_array[j+1]=dtemp;
				
				memcpy(itemp,genome[j],genome_size*sizeof(int));
				memcpy(genome[j],genome[j+1],genome_size*sizeof(int));
				memcpy(genome[j+1],itemp,genome_size*sizeof(int));
			}
		}
	}
}

/* Crossover operation: based on tournament selection */
/* Select the tournament_size based on the genome count : */
/*     (if genome_count > 100 ) tournament_size = 10   else   tournament_size = 4 */
/* Select 2 chromosomes based on the tournament size and cross them over based on the crossover probability */
/* There is 1 crossover point and it is random */
void	Population::crossover()
{
        int parent[2];
        int nchildren=(int)((1.0 - selection_rate) * genome_count);
	if(!(nchildren%2==0)) nchildren++;
        const int tournament_size =(genome_count<=100)?4:10;
        int count_children=0;
        while(1)
        {
		// The two parents are selected here according to the tournament selection procedure
                for(int i=0;i<2;i++)
                {
                        double max_fitness=-1e+10;
                        int    max_index=-1;
			int r;
			// Select the best parents of  the candidates 
                        for(int j=0;j<tournament_size;j++)
                        {
				r=rand() % (genome_count);
                                if(j==0 || fitness_array[r]>max_fitness)
                                {
                                        max_index=r;
                                        max_fitness=fitness_array[r];
                                }
                        }
                        parent[i]=max_index;
                }
        int pt1;
		// The one-point crossover is performed here (the point is pt1)
		pt1=rand() % genome_size;
		memcpy(children[count_children],
				genome[parent[0]],pt1 * sizeof(int));
		memcpy(&children[count_children][pt1],
			&genome[parent[1]][pt1],(genome_size-pt1)*sizeof(int));
		memcpy(children[count_children+1],
				genome[parent[1]],pt1 * sizeof(int));
		memcpy(&children[count_children+1][pt1],
			&genome[parent[0]][pt1],(genome_size-pt1)*sizeof(int));
		count_children+=2;
		if(count_children>=nchildren) break;
	}
	
	

	for(int i=0;i<nchildren;i++)
	{
		memcpy(genome[genome_count-i-1],
				children[i],genome_size * sizeof(int));
	}
}

void	Population::setElitism(int s)
{
	elitism = s;
}

/* Mutate the current population */
/* Standard mutation algorithm: mutate all chromosomes in the population based on the mutation probability */
void	Population::mutate()
{
	int start = elitism * (int)(genome_count*selection_rate);
	
	start = elitism;
	start = 1;

	for(int i=start;i<genome_count;i++)
	{
		for(int j=0;j<genome_size;j++)
		{
			double r=rand()*1.0/RAND_MAX;
			if(r<mutation_rate)
			{
				genome[i][j]=rand() % MAX_RULE;
			}
		}
	}
}

/* Evaluate the fitness for all chromosomes in the current population */
void	Population::calcFitnessArray()
{
	vector<int> g;
	g.resize(genome_size);

	double dmin = 1e+100;
	for(int i=0;i<genome_count;i++)
	{
		for(int j=0;j<genome_size;j++) g[j]=genome[i][j];	
			fitness_array[i]=fitness(g);	
		if(fabs(fitness_array[i])<dmin) dmin=fabs(fitness_array[i]);
/*		if(i%10==0)
		{
			printf("%d:%lf ",i,dmin);
			fflush(stdout);
		}*/
	}
//	printf("\n");
}

/* Return the current generation */
int	Population::getGeneration() const
{
	return generation;
}

/* Return the genome count */
int	Population::getCount() const
{
	return genome_count;
}

/* Return the size of the population */
int	Population::getSize() const
{
	return genome_size;
}

/* Evolve the next generation */
void	Population::nextGeneration()
{
	if(generation==0)
	calcFitnessArray();
	select();
	crossover();
	mutate();
	calcFitnessArray();
    extern int localSearchGenerations,localSearchChromosomes;
    if((generation+1)%100==0)
        for(int i=0;i<20;i++)
		localSearch(rand() % genome_count);
	
	++generation;
}

void	Population::localSearch(int gpos)
{
	vector<int> g;
	g.resize(genome_size);
	int pos=gpos;
	for(int iters=1;iters<=100;iters++)
	{
		int randgenome=rand() % genome_count;
		int randpos=rand() % genome_size;
		for(int i=0;i<randpos;i++) g[i]=genome[pos][i];
		for(int i=randpos;i<genome_size;i++) g[i]=genome[randgenome][i];
		double f=fitness(g);
		if(fabs(f)<fabs(fitness_array[pos]))
		{
			for(int i=0;i<genome_size;i++) genome[pos][i]=g[i];
			fitness_array[pos]=f;
		}
		else
		{
			for(int i=0;i<randpos;i++) g[i]=genome[randgenome][i];
			for(int i=randpos;i<genome_size;i++) g[i]=genome[pos][i];
			f=fitness(g);
			if(fabs(f)<fabs(fitness_array[pos]))
			{
				for(int i=0;i<genome_size;i++) genome[pos][i]=g[i];
				fitness_array[pos]=f;
			}
		}
	}
	return;
}

/* Set the mutation rate */
void	Population::setMutationRate(double r)
{
	if(r>=0 && r<=1) mutation_rate = r;
}

/* Set the selection rate */
void	Population::setSelectionRate(double r)
{
	if(r>=0 && r<=1) selection_rate = r;
}

/* Return the selection rate */
double	Population::getSelectionRate() const
{
	return selection_rate;
}

/* Return the mutation rate */
double	Population::getMutationRate() const
{
	return mutation_rate;
}

/* Return the best fitness for this population */
double	Population::getBestFitness() const
{
	return  fitness_array[0];
}

/* Return the best chromosome */
vector<int> Population::getBestGenome() const
{
	vector<int> g;g.resize(genome_size);
	for(int i=0;i<genome_size;i++) g[i]=genome[0][i];
	return g;
}

/* Evaluate and return the best fitness for all chromosomes in the population */
double	Population::evaluateBestFitness() 
{
	vector<int> g;g.resize(genome_size);
	for(int i=0;i<genome_size;i++) g[i]=genome[0][i];	
	return fitness(g);
}

void	Population::setBest(vector<int> &g,double f)
{
	double tf=fitness(g);
	if(tf<fitness_array[0] && fabs(tf-f)>1e-4) 
	{
		return;
	}
	if(g.size()>genome_size)
	{
		int *old=new int[genome_size];
		for(int i=0;i<genome_count;i++)
		{
			for(int j=0;j<genome_size;j++) old[j]=genome[i][j];
			delete[] genome[i];
			genome[i]=new int[g.size()];
			for(int j=0;j<g.size();j++) 	genome[i][j]=0;
			for(int j=0;j<genome_size;j++) genome[i][j]=old[j];
			delete[] children[i];
			children[i]=new int[g.size()];
		}
		genome_size=g.size();
		extern int genome_length;
		genome_length=genome_size;
		delete[] old;
	}
    int pos=0;
	for(int i=0;i<genome_size;i++) 
	{
		genome[pos][i]=g[i];	
	}
	fitness_array[pos]=f;
}

void	Population::setGenome(int pos,vector<int> &g,double f,int k)
{
	double tf=fitness(g);
	if(tf<fitness_array[pos] && fabs(tf-f)>1e-4) return;
	if(g.size()>genome_size)
	{
		int *old=new int[genome_size];
		for(int i=0;i<genome_count;i++)
		{
			for(int j=0;j<genome_size;j++) old[j]=genome[i][j];
			delete[] genome[i];
			genome[i]=new int[g.size()];
			for(int j=0;j<g.size();j++) genome[i][j]=0;
			int ipos=0;
			for(int l=0;l<k;l++)
			{
				for(int j=0;j<genome_size/k;j++)
				{
					genome[i][l*g.size()/k+j]=old[ipos];
					ipos++;
				}
			}
			delete[] children[i];
			children[i]=new int[g.size()];
		}
		genome_size=g.size();
		extern int genome_length;
		genome_length=genome_size;
		delete[] old;
	}
	for(int i=0;i<genome_size;i++) 
	{
		genome[pos][i]=g[i];	
	}
	fitness_array[pos]=f;
}

void	Population::setGenome(int pos,vector<int> &g,double f)
{
	double tf=fitness(g);
	if(tf<fitness_array[pos] && fabs(tf-f)>1e-4) return;
	if(g.size()>genome_size)
	{
		int *old=new int[genome_size];
		for(int i=0;i<genome_count;i++)
		{
			for(int j=0;j<genome_size;j++) old[j]=genome[i][j];
			delete[] genome[i];
			genome[i]=new int[g.size()];
			for(int j=0;j<g.size();j++) genome[i][j]=0;
			for(int j=0;j<genome_size;j++) genome[i][j]=old[j];
			delete[] children[i];
			children[i]=new int[g.size()];
		}
		genome_size=g.size();
		extern int genome_length;
		genome_length=genome_size;
		delete[] old;
	}
	for(int i=0;i<genome_size;i++) 
	{
		genome[pos][i]=g[i];	
	}
	fitness_array[pos]=f;
}

void	Population::getGenome(int pos,vector<int> &g)
{
	g.resize(genome_size);
	for(int i=0;i<g.size();i++) g[i]=genome[pos][i];
}


/* Destructor */
Population::~Population()
{
	for(int i=0;i<genome_count;i++)
	{
		delete[] children[i];
		delete[] genome[i];
	}
	delete[] genome;
	delete[] children;
	delete[] fitness_array;
}
