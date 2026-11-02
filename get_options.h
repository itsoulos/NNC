# ifndef __GETOPTIONS__H
# define __GETOPTIONS__H
# include <QString>
extern void print_usage();
extern int random_seed;
extern QString fitness_metric;
extern int genome_count;
extern int genome_length;
extern double selection_rate;
extern double mutation_rate;
extern  QString   train_file;
extern  QString    test_file;
extern  void	parse_cmd_line(int argc,char **argv);
extern	QString  kind;//neural, ode, pde, sode, kdv
extern	int 	generations;
extern  QString  output_file;
extern int localSearchGenerations;
extern int localSearchChromosomes;
# endif
