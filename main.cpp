#include <QCoreApplication>
# include <get_options.h>
# include <nncneuralprogram.h>
# include <population.h>
# include <tolmin.h>
# include <converter.h>
# include <odeneuralprogram.h>
# include <pdeneuralprogram.h>
# include <sodeneuralprogram.h>
# include <kdvneuralprogram.h>
# include <QFile>
# include <QTextStream>
# include <QIODevice>
NeuralProgram *program=NULL;
Population *pop=NULL;

typedef vector<int>Genome;
int getDimension(QString filename)
{
    int d=0;
    QFile fp(filename);
    if(!fp.open(QIODevice::ReadOnly|QIODevice::Text)) return 0;
    QTextStream st(&fp);
    st>>d;
    fp.close();
    return d;
}

void run()
{
    int d=0;
    if(kind=="neural")
    {
        d=getDimension(train_file);
        program=new NNCNeuralProgram (d,train_file,test_file);
    }
    else
    if(kind=="ode")
    {
        d=1;
        program=new OdeNeuralProgram(train_file);
    }
    else
    if(kind=="pde")
    {
        d=2;
        program=new PdeNeuralProgram(train_file);
    }
    else
    if(kind=="kdv")
    {
        d=2;
        program=new KdvNeuralProgram(train_file);
    }

    pop=new Population (genome_count,genome_length ,program);
    pop->setSelectionRate(selection_rate);
    pop->setMutationRate(mutation_rate);
    const int max_generations=generations;
    vector<int> genome;
    genome.resize(genome_length);
    string str;
    double f;
    double old_test_error=0.0;
    Data bestWeights;
    vector<int> bestGenome;
    bool hasBest=false;
    double bestError=1e+100;
    double bestFitness=-1e+100;
    for(int i=1;i<=max_generations;i++)
    {
            pop->nextGeneration();
            f=pop->getBestFitness();
            genome=pop->getBestGenome();
            str=program->printProgram(genome);
            program->fitness(genome);
            if(fitness_metric=="sse" ? fabs(f)<bestError : f>bestFitness)
            {
                program->neuralparser->getWeights(bestWeights);

                bestError=fabs(f);
                bestFitness=f;
                bestGenome=genome;
                hasBest=true;
                old_test_error=program->getTestError();

            }
	    printf("%d\t%lf\t%s\n",i,f,str.c_str());
            //printf("BEST[%d]=%20.10lf Solution: y(x)=%s\n",i,f,str.c_str());
    //LOCALSEARCH
            if(i%localSearchGenerations==0)
            {
                int imax=localSearchChromosomes;
                int iflag=0;
                for(int k=0;k<imax;k++)
                {
                    vector<int> trial_genome;
                    trial_genome.resize(genome_length);
                    int trial_pos;
                    Data x;
    again:
                    if(iflag==0)
                    trial_pos=rand() % genome_count;//(rand()%2==1)?0:(rand() % genome_count);//genome_count-1;
                    else
                    trial_pos=rand() % genome_count;
                    iflag=1;

                    pop->getGenome(trial_pos,trial_genome);
                    program->fitness(trial_genome);
                    program->neuralparser->getWeights(x);
                    double value=0;
                    MinInfo Info1;
                    Info1.iters=2001;
                    Info1.problem=program;
                    double old_f=1e+100;
                    int tries=0;
                    do
                    {
                        value=program->getTrainError();
                        if(value>=1e+8) {iflag=1;goto again;}
                        value=tolmin(x,Info1);
                        if(value>=1e+8) {iflag=1;goto again;}
                        program->neuralparser->getWeights(x);
                        if(fabs(old_f-value)<1e-5) break;
                        old_f=value;

                        fflush(stdout);
                        tries++;
                        if(tries>=20) break;
                        break;
                    }while(1);

                    program->neuralparser->getWeights(x);
                    value=program->getTrainError();

                    if((std::isnan(value) || std::isinf(value)))
                    {
                         if(!k) continue;
                        iflag=1;
                         goto again;
                    }

                    Converter con(x,x.size()/(d+2),d);
                    con.convert(trial_genome);
                    for(int i=0;i<trial_genome.size();i++)
                    {
                        //if(abs(trial_genome[i])>255) trial_genome[i]=0;
                    }
                    double trial_fitness=program->fitness(trial_genome);
                    // Do not replace a chromosome when local optimization worsens
                    // the selected evolutionary fitness.
                    vector<int> previous;
                    pop->getGenome(trial_pos,previous);
                    const double previous_fitness=program->fitness(previous);
                    if(std::isfinite(trial_fitness) && trial_fitness>previous_fitness) {
                        pop->setGenome(trial_pos,trial_genome,trial_fitness);
                        if(!hasBest || trial_fitness>bestFitness) {
                            bestFitness=trial_fitness;
                            bestGenome=trial_genome;
                            hasBest=true;
                            program->fitness(trial_genome);
                            program->neuralparser->getWeights(bestWeights);
                            bestError=program->getTrainError();
                            old_test_error=program->getTestError();
                            pop->setBest(trial_genome,trial_fitness);
                        }
                    }
                }
                pop->select();
            }
          if(fitness_metric=="sse" && fabs(bestError)<1e-6) break;
        }
        if(hasBest) genome=bestGenome;
        if(hasBest) program->fitness(genome);
        const double final_fitness=hasBest ? program->fitness(genome) : -1e+8;
        const double train_sse=program->getTrainError();
        const double test_sse=program->getTestError();
        str=program->printProgram(genome);
        printf("TRAIN FITNESS = %.10lf (%s; higher is better)\n",final_fitness,fitness_metric.toStdString().c_str());
        printf("TRAIN SSE = %.10lf\n",train_sse);
        printf("TEST SSE = %.10lf\n",test_sse);
        if(kind=="neural" && fitness_metric!="sse") {
            NNCNeuralProgram *p=(NNCNeuralProgram*)program;
            p->printClassificationMetrics(genome,false);
            if(fitness_metric=="macro_f1") {
                const double check=p->getClassificationFitness();
                printf("TRAIN FITNESS CHECK = %.10lf (difference from reported fitness: %.10g)\n",check,fabs(check-final_fitness));
            }
            p->printClassificationMetrics(genome,true);
        } else if(kind=="neural") {
            NNCNeuralProgram *p=(NNCNeuralProgram*)program;
            p->printClassificationMetrics(genome,false);
            if(fitness_metric=="macro_f1") {
                const double check=p->getClassificationFitness();
                printf("TRAIN FITNESS CHECK = %.10lf (difference from reported fitness: %.10g)\n",check,fabs(check-final_fitness));
            }
            p->printClassificationMetrics(genome,true);
        }
        printf("SOLUTION: y(x)=%s\n",str.c_str());
        if(output_file!="") program->printOutput(output_file);
        delete pop;
        delete program;
}


void runSode()
{
        SodeNeuralProgram p(train_file);
        genome_length = genome_length * p.getNode();
        Population pop(genome_count,genome_length ,&p);
        pop.setSelectionRate(selection_rate);
        pop.setMutationRate(mutation_rate);
        const int max_generations=generations;
        vector<int> genome;
        genome.resize(genome_length);
        string str;
        double f;
        double old_test_error=0.0;

        vector<Data> bestWeights;
        bestWeights.resize(p.getNode());
        double bestError=1e+100;
        for(int i=1;i<=max_generations;i++)
        {
            pop.nextGeneration();
            genome=pop.getBestGenome();
            p.fitness(genome);
            str=p.printProgram(genome);
            f=p.getTrainError();
            if(fabs(f)<bestError)
            {
                for(int j=0;j<p.getNode();j++)
                    p.nparser[j]->getWeights(bestWeights[j]);
                bestError=fabs(f);
                old_test_error=p.getTestError();
            }
            printf("BEST[%d]=%20.10lf\n",i,f);
    //LOCALSEARCH

            vector<double> fvalue;
            fvalue.resize(p.getNode());
            if(i%localSearchGenerations==0)
            {
                for(int k=0;k<localSearchChromosomes;k++)
                {
                    vector<int> trial_genome;
                    trial_genome.resize(genome_length);
                    int trial_pos;
                    vector<Data> x;
                    x.resize(p.getNode());

                    if(i>localSearchChromosomes && k==0)
                    {
                        trial_pos=0;
                        for(int j=0;j<p.getNode();j++)
                        {
                            p.nparser[j]->setWeights(bestWeights[j]);
                            x[j].resize(bestWeights[j].size());
                            x[j]=bestWeights[j];
                        }
                    }
                    else
                    {
                        trial_pos=rand() % genome_count;
                        pop.getGenome(trial_pos,trial_genome);
                        p.fitness(trial_genome);
                        for(int j=0;j<p.getNode();j++)
                        {
                            p.nparser[j]->getWeights(x[j]);
                        }
                    }

                    MinInfo Info1;
                    Info1.problem=&p;
                    Info1.iters=2001;
                    double old_value=p.getTrainError();
                    double value;
                    int ik=0;
                    do{
                    for(int j=0;j<p.getNode();j++)
                    {
                        p.neuralparser=p.nparser[j];
                        p.currentparser=j;
                        double old_f=p.getTrainError();
                        int tries=0;
                        do{

                            p.neuralparser->getWeights(x[j]);


                            value=tolmin(x[j],Info1);
                            fvalue[j]=-value;

                            if(fabs(value-old_f)<1e-5) break;
                            old_f=value;


                            fflush(stdout);

                            tries++;
                            if(tries>=20) break;
                        }while(1);
                    }
                    double new_value=p.getTrainError();
                    if(fabs(new_value-old_value)<1e-5) break;
                    ik++;
                    if(ik>=1) break;
                    }while(1);

    //END LOCAL
                    for(int j=0;j<p.getNode();j++) p.nparser[j]->getWeights(x[j]);
                    value=p.getTrainError();

                    if((std::isnan(value) || std::isinf(value)))
                    {
                        continue;
                    }

                    trial_genome.resize(0);
                    vector<Genome> subgenome;
                    subgenome.resize(p.getNode());

                    int max_length=0;
                    for(int j=0;j<p.getNode();j++)
                    {
                        int d=1;
                        Converter con(x[j],x[j].size()/(d+2),d);
                        subgenome[j].resize(genome_length/p.getNode());
                        con.convert(subgenome[j]);
                        if(subgenome[j].size()>max_length) max_length=subgenome[j].size();
                    }
                    for(int j=0;j<p.getNode();j++)
                    {
                        int s=subgenome[j].size();
                        if(s<max_length)
                        {
                            subgenome[j].resize(max_length);
                            for(int k=s;k<max_length;k++) subgenome[j][k]=0;
                        }
                        for(int k=0;k<subgenome[j].size();k++) trial_genome.push_back(subgenome[j][k]);
                    }

                    for(int i=0;i<trial_genome.size();i++)
                    {
                        if(abs(trial_genome[i])>255) trial_genome[i]=0;
                    }

                    double trial_fitness=-value;
                    pop.setGenome(trial_pos,trial_genome,trial_fitness,p.getNode());

                    if(fabs(value)<=bestError)
                    {
                        for(int j=0;j<p.getNode();j++)
                        {
                            bestWeights[j]=x[j];
                            p.nparser[j]->setWeights(x[j]);
                        }
                        bestError=fabs(value);
                        old_test_error=p.getTestError();
                        pop.setGenome(0,trial_genome,trial_fitness,p.getNode());
                        f=trial_fitness;
                    }

                    if(value<f)
                    {
                        for(int j=0;j<p.getNode();j++)
                        {
                            bestWeights[j]=x[j];
                            p.nparser[j]->setWeights(x[j]);
                        }
                        bestError=fabs(value);
                        old_test_error=p.getTestError();
                        pop.setGenome(0,trial_genome,trial_fitness,p.getNode());
                        f=trial_fitness;
                    }


                }
                pop.select();
            }

            if(fabs(bestError)<1e-6) break;
        }
        for(int j=0;j<p.getNode();j++)
        p.nparser[j]->setWeights(bestWeights[j]);
        old_test_error=p.getTestError();
        printf("TRAIN ERROR =%.10lf\n",bestError);
        printf("TEST  ERROR =%.10lf\n",old_test_error);
        if(output_file!="") p.printOutput(output_file);
}

int main(int argc, char *argv[])
{
    parse_cmd_line(argc,argv);
    if(argc==1) print_usage();
    srand(random_seed);
    if(kind=="sode")
        runSode();
    else
        run();
    return 0;
}
