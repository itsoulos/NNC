# include <nncneuralprogram.h>
# include <math.h>
# include <get_options.h>
# include <algorithm>
# include <limits>
# include <QFile>
# include <QTextStream>
NNCNeuralProgram::NNCNeuralProgram(int Dimension,QString TrainFile,QString TestFile):
	NeuralProgram(Dimension)
{

	isvalidation=0;
    char cfile1[1024],cfile2[1024];
    strcpy(cfile1,TrainFile.toStdString().c_str());
    strcpy(cfile2,TestFile.toStdString().c_str());
	if(TrainFile!=NULL)
	{
        FILE *fp=fopen(cfile1,"r");
		if(!fp)  exit(EXIT_FAILURE);
		int d;
		fscanf(fp,"%d",&dimension);
		xtemp=new double[dimension];
		int tcount;
		fscanf(fp,"%d",&tcount);
		if(tcount<=0) {fclose(fp); return ;}
		train_xpoint.resize(tcount);
		train_ypoint.resize(tcount);
		xmax.resize(dimension);
		xmin.resize(dimension);
		for(int i=0;i<tcount;i++)
		{
			train_xpoint[i].resize(dimension);
			for(int j=0;j<dimension;j++) 
			{
				fscanf(fp,"%lf",&train_xpoint[i][j]);	
				if(i==0 || train_xpoint[i][j]>xmax[j]) xmax[j]=train_xpoint[i][j];
				if(i==0 || train_xpoint[i][j]<xmin[j]) xmin[j]=train_xpoint[i][j];
			}
			fscanf(fp,"%lf",&train_ypoint[i]);
		}

        // Class labels must be learned from training data only.
        categ.clear();
        for(size_t i=0;i<train_ypoint.size();++i) {
            bool found=false;
            for(size_t j=0;j<categ.size();++j)
                if(fabs(categ[j]-train_ypoint[i])<1e-7) {found=true;break;}
            if(!found) categ.push_back(train_ypoint[i]);
        }
		fclose(fp);
	}
    if(TestFile!="")
	{
        FILE *fp=fopen(cfile2,"r");
		if(!fp) return;
		int d;
		fscanf(fp,"%d",&d);
		if(d!=dimension) {fclose(fp); return ;}
		int tcount;
		fscanf(fp,"%d",&tcount);
		if(tcount<=0) {fclose(fp); return ;}
		test_xpoint.resize(tcount);
		test_ypoint.resize(tcount);
		for(int i=0;i<tcount;i++)
		{
			test_xpoint[i].resize(dimension);
			for(int j=0;j<dimension;j++)
				fscanf(fp,"%lf",&test_xpoint[i][j]);
			fscanf(fp,"%lf",&test_ypoint[i]);

		}
		fclose(fp);
	}
	program=new SigProgram(dimension);
	setStartSymbol(program->getStartSymbol());
	neuralparser=new NeuralParser(dimension);

}

static void mymap(Data x,Data &x1)
{
	for(int i=0;i<x.size();i++)
		x1[i]=x[i];
}


void	NNCNeuralProgram::getDeriv(Data &g)
{
	for(int i=0;i<g.size();i++) g[i]=0.0;
	Data tempg;
	tempg.resize(g.size());
	int start=0;
	int end=train_ypoint.size();
	if(isvalidation) end=4*train_ypoint.size()/5;
	for(int i=start;i<end;i++)
	{
		double v=neuralparser->eval(train_xpoint[i])-train_ypoint[i];
		neuralparser->getDeriv(train_xpoint[i],tempg);
		for(int j=0;j<g.size();j++) g[j]+=2.0*v*tempg[j];
	}
}

double	NNCNeuralProgram::getTestError()
{
	double value=0.0;
    double *xx=new double[dimension];

	for(int i=0;i<test_ypoint.size();i++)
	{
        	for(int j=0;j<dimension;j++) xx[j]=test_xpoint[i][j];
		if(program!=NULL)
		{
        	double v=neuralparser->eval(xx)-test_ypoint[i];
		value=value+v*v;
		}
	}
	delete[] xx;
	return value;
}

int	NNCNeuralProgram::getTrainSize() const
{
	return train_ypoint.size();
}

int	NNCNeuralProgram::getTestSize() const
{
	return test_ypoint.size();
}


double	NNCNeuralProgram::getTrainError()
{
	double value=0.0;
	Data xx;
	xx.resize(dimension);

	int start=0;
	int end=train_ypoint.size();
	if(isvalidation)
	{		
		start=0;
		end=4*train_ypoint.size()/5;
	}
	for(int i=start;i<end;i++)
	{
		for(int j=0;j<dimension;j++) xtemp[j]=train_xpoint[i][j];
		double v=neuralparser->eval(xtemp);
        if(program->EvalError() || std::isnan(v) || std::isinf(v))
		{
			return 1e+8;
		}
		v=(v-train_ypoint[i]);
		value=value+v*v;
	}
    return value;
}

double NNCNeuralProgram::getClassificationFitness()
{
    const int n=(int)categ.size();
    if(n<2 || train_ypoint.empty()) return -1e+8;
    std::vector<double> tp(n,0.0), actual(n,0.0), predicted(n,0.0);
    double correct=0.0, total=0.0;
    const int end=isvalidation ? 4*(int)train_ypoint.size()/5 : (int)train_ypoint.size();
    for(int i=0;i<end;i++) {
        double v=neuralparser->eval(train_xpoint[i].data());
        if(!std::isfinite(v) || program->EvalError()) return -1e+8;
        int truth=-1, guess=0;
        double dist=std::numeric_limits<double>::infinity();
        for(int j=0;j<n;j++) {
            if(fabs(train_ypoint[i]-categ[j])<1e-7) truth=j;
            double delta=fabs(v-categ[j]);
            if(delta<dist) {dist=delta;guess=j;}
        }
        if(truth<0) return -1e+8;
        actual[truth]++; predicted[guess]++; total++;
        if(truth==guess) {tp[truth]++;correct++;}
    }
    if(total==0) return -1e+8;
    double macroRecall=0, macroF1=0, weightedF1=0, logRecall=0;
    int present=0;
    for(int j=0;j<n;j++) {
        if(actual[j]==0) continue;
        present++;
        const double recall=tp[j]/actual[j];
        const double precision=predicted[j]>0 ? tp[j]/predicted[j] : 0.0;
        const double f1=precision+recall>0 ? 2*precision*recall/(precision+recall) : 0.0;
        macroRecall+=recall; macroF1+=f1;
        weightedF1+=actual[j]*f1/total;
        if(recall==0) logRecall=-std::numeric_limits<double>::infinity();
        else if(std::isfinite(logRecall)) logRecall+=log(recall);
    }
    macroRecall/=present; macroF1/=present;
    if(fitness_metric=="error") return correct/total-1.0;
    if(fitness_metric=="balanced_error") return macroRecall-1.0;
    if(fitness_metric=="macro_recall") return macroRecall;
    if(fitness_metric=="macro_f1") return macroF1;
    if(fitness_metric=="weighted_f1") return weightedF1;
    if(fitness_metric=="gmean") return std::isfinite(logRecall) ? exp(logRecall/present) : 0.0;
    if(fitness_metric=="mcc") {
        double sumPT=0, sumP2=0, sumT2=0;
        for(int j=0;j<n;j++) {
            sumPT+=predicted[j]*actual[j];
            sumP2+=predicted[j]*predicted[j];
            sumT2+=actual[j]*actual[j];
        }
        double den=sqrt((total*total-sumP2)*(total*total-sumT2));
        return den>0 ? (correct*total-sumPT)/den : 0.0;
    }
    return -1e+8;
}

void NNCNeuralProgram::printClassificationMetrics(vector<int> &genome, bool test)
{
    const vector<Data> &xs = test ? test_xpoint : train_xpoint;
    const Data &ys = test ? test_ypoint : train_ypoint;
    const char *prefix = test ? "TEST" : "TRAIN";
    if(xs.empty() || categ.empty()) {
        printf("%s METRICS unavailable (empty dataset or classes)\n", prefix);
        return;
    }
    string expression;
    if(!getElements(genome, expression)) {
        printf("%s METRICS unavailable (invalid genome)\n", prefix);
        return;
    }
    program->Parse(expression);
    const int n=(int)categ.size();
    vector<double> tp(n,0), actual(n,0), predicted(n,0);
    std::vector<std::vector<int>> confusion(n,std::vector<int>(n,0));
    double correct=0, invalid=0;
    for(size_t i=0;i<ys.size();++i) {
        int truth=-1;
        for(int j=0;j<n;++j) if(fabs(ys[i]-categ[j])<1e-7) {truth=j;break;}
        if(truth<0) {++invalid;continue;}
        actual[truth]++;
        double v=program->Eval(xs[i].data());
        if(!std::isfinite(v) || program->EvalError()) {++invalid;continue;}
        int guess=0;
        for(int j=1;j<n;++j) if(fabs(v-categ[j])<fabs(v-categ[guess])) guess=j;
        predicted[guess]++;
        confusion[truth][guess]++;
        if(guess==truth) {tp[truth]++;correct++;}
    }
    double recallSum=0, precisionSum=0, weightedRecall=0, weightedPrecision=0, f1Sum=0, weightedF1=0, logRecall=0;
    int present=0; bool zeroRecall=false;
    for(int j=0;j<n;++j) {
        if(actual[j]==0) continue;
        ++present;
        double recall=tp[j]/actual[j];
        double precision=predicted[j]>0 ? tp[j]/predicted[j] : 0;
        double f1=precision+recall>0 ? 2*precision*recall/(precision+recall) : 0;
        recallSum+=recall;precisionSum+=precision;f1Sum+=f1;
        weightedPrecision+=actual[j]*precision/ys.size();
        weightedRecall+=actual[j]*recall/ys.size();
        weightedF1+=actual[j]*f1/ys.size();
        if(recall<=0) zeroRecall=true; else logRecall+=log(recall);
    }
    if(!present) {printf("%s METRICS unavailable (no known labels)\n",prefix);return;}
    double sumPT=0,sumP2=0,sumT2=0;
    for(int j=0;j<n;++j) {
        sumPT+=predicted[j]*actual[j];
        sumP2+=predicted[j]*predicted[j];
        sumT2+=actual[j]*actual[j];
    }
    const double total=(double)ys.size();
    const double den=sqrt(std::max(0.0,(total*total-sumP2)*(total*total-sumT2)));
    const double mcc=den>0 ? (correct*total-sumPT)/den : 0;
    printf("%s CLASS ERROR = %.6f%%\n",prefix,100*(1-correct/total));
    printf("%s ACCURACY = %.10f\n",prefix,correct/total);
    printf("%s BALANCED ACC = %.10f\n",prefix,recallSum/present);
    printf("%s MACRO PRECISION = %.10f\n",prefix,precisionSum/present);
    printf("%s WEIGHTED PRECISION = %.10f\n",prefix,weightedPrecision);
    printf("%s MACRO RECALL = %.10f\n",prefix,recallSum/present);
    printf("%s WEIGHTED RECALL = %.10f\n",prefix,weightedRecall);
    printf("%s MACRO F1 = %.10f\n",prefix,f1Sum/present);
    printf("%s WEIGHTED F1 = %.10f\n",prefix,weightedF1);
    printf("%s GMEAN = %.10f\n",prefix,zeroRecall ? 0 : exp(logRecall/present));
    printf("%s MCC = %.10f\n",prefix,mcc);
    if(invalid) printf("%s INVALID/UNKNOWN = %.0f\n",prefix,invalid);
    printf("%s CONFUSION MATRIX (rows=actual, columns=predicted):\n",prefix);
    printf("%s LABELS:",prefix);
    for(int j=0;j<n;++j) printf(" %.10g",categ[j]);
    printf("\n");
    for(int j=0;j<n;++j) {
        printf("%s CM %.10g:",prefix,categ[j]);
        for(int k=0;k<n;++k) printf(" %d",confusion[j][k]);
        printf("\n");
    }
}

double	NNCNeuralProgram::getClassTestError(vector<int> &genome)
{
	double value=0.0;
	string str;
	if(!getElements(genome,str)) return -1e+8;
	program->Parse(str);
	double *xx=new double[train_xpoint[0].size()];

	for(int i=0;i<test_ypoint.size();i++)
	{
		for(int j=0;j<test_xpoint[i].size();j++) xx[j]=test_xpoint[i][j];
		double v=program->Eval(xx);
        if(program->EvalError() || std::isnan(v) || std::isinf(v))
		{
			value+=1.0;
			continue;
		}
		
		double minValue=1e+10;
		int index=-1;
		for(int j=0;j<categ.size();j++)
		{
			if(fabs(categ[j]-v)<minValue)
			{
				minValue=fabs(categ[j]-v);
				index = j;
			}
		}
		if(index<0) {value+=1.0;continue;}
		double myclass=categ[index];
		value+=fabs(test_ypoint[i]-myclass)>1e-5;
	}

	delete[] xx;
	return value*100.0/test_ypoint.size();
}

void     NNCNeuralProgram::printOutput(QString filename)
{
    QFile fp(filename);
    if(!fp.open(QIODevice::WriteOnly |QIODevice::Text)) return;
    QTextStream st(&fp);
    double *xx=new double[train_xpoint[0].size()];
    for(int i=0;i<test_ypoint.size();i++)
    {
        for(int j=0;j<dimension;j++) st<<test_xpoint[i][j]<<" ";
        for(int j=0;j<test_xpoint[i].size();j++) xx[j]=test_xpoint[i][j];
        double v=program->Eval(xx);
        st<<test_ypoint[i]<<" "<<v<<endl;
    }
    delete[] xx;
    fp.close();
}

NNCNeuralProgram::~NNCNeuralProgram()
{
	delete[] xtemp;
}
