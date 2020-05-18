package milestone.due.engine;

import com.opencsv.CSVWriter;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;


public class WekaEngine {

    /**
     * Questa classe ha lo scopo di andare a calcolare, attraverso il metodo Walk Forward
     * Le statistiche. In particolare il numero di run da effettuare sarà pari al numero
     * delle parti in cui è diviso il file - 1 --> eg: Bookeeper avrà n=6 in quanto le release
     * sono 7 in totale
     * <p>
     * Una volta eseguite tutte le run verrà fatta la media dei risultati.
     * <p>
     * Classificatori --> RandomForest, NaiveBayes, IBK
     */

    private static final Logger LOGGER = Logger.getLogger(WekaEngine.class.getName());

    static String m1d2Test = "";
    static String m1d2Train = "";
    static String prefix = "";
    static String out = "";
    static String numRelease = "";

    private static void importResources(int value) {
        /**
         * Attraverso config.properties andiamo a caricare i valori delle stringhe per le open e le write dei file.
         * Necessario al fine di evitare copie inutili dello stesso codice in locazioni diverse della classe.
         */
        try (InputStream input = new FileInputStream("C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\config.properties")) {

            Properties prop = new Properties();
            // load a properties file
            prop.load(input);


            if (value == 0) {
                m1d2Test = prop.getProperty("BOOKARFFTESTING");
                m1d2Train = prop.getProperty("BOOKARFFTRAINING");
                prefix = prop.getProperty("prefixBOOK");
                out = prop.getProperty("OUTBOOK");
                numRelease = prop.getProperty("NUMBOOK");
            } else {
                m1d2Test = prop.getProperty("TAJOARFFTESTING");
                m1d2Train = prop.getProperty("TAJOARFFTRAINING");
                prefix = prop.getProperty("prefixTAJO");
                out = prop.getProperty("OUTTAJO");
                numRelease = prop.getProperty("NUMTAJO");

            }

        } catch (IOException e) {
            LOGGER.log(Level.WARNING, String.valueOf(e));
        }
    }


    public List<String[]> walkForwardValidation(int numOfSteps) throws Exception {

        /**
         * Con questo metodo andiamo ad applicare la tecnica Walk Forward come tecnica di validazione
         * per il nostro dataset, diviso in base alle release.
         *
         * @param numOfSteps: Int --> Numero di passi per la validazione.
         */
        String tst = "";
        String trn = "";
        int lock = 0;

        List<String[]> ret = new ArrayList<>();
        ret.add(new String[]{"Dataset", "#TrainingRelease", "Classifier", "Precision", "Recall", "AUC", "Kappa"});

        for (int i = 1; i <= numOfSteps - 1; i++) {
            tst = m1d2Test + "\\M1D2" + prefix + i + "testing.arff";
            trn = m1d2Train + "\\M1D2" + prefix + i + "training.arff";
            ConverterUtils.DataSource ts = new ConverterUtils.DataSource(tst);
            if (lock == 0) {

                new WekaEngine().calculateRandomForest(null, ts, 0, i, ret);
                new WekaEngine().calculateNaiveBayes(null, ts, 0, i, ret);
                new WekaEngine().calculateIBK(null, ts, 0, i, ret);

            } else {

                ConverterUtils.DataSource tr = new ConverterUtils.DataSource(trn);
                new WekaEngine().calculateRandomForest(tr, ts, 1, i, ret);
                new WekaEngine().calculateNaiveBayes(tr, ts, 1, i, ret);
                new WekaEngine().calculateIBK(tr, ts, 1, i, ret);

            }
            lock = 1;

        }

        return ret;
    }

    private void calculateRandomForest(ConverterUtils.DataSource train, ConverterUtils.DataSource test, int value, int counter, List<String[]> list) throws Exception {
        /**
         * Value va a determinare se è solo l'istanza di training o anche testing.
         */

        Instances testing = test.getDataSet();
        RandomForest classifier = new RandomForest();

        if (value == 0) {

            int numAttr = testing.numAttributes();
            testing.setClassIndex(numAttr - 1);
            classifier.buildClassifier(testing);

            Evaluation eval = new Evaluation(testing);
            eval.evaluateModel(classifier, testing);

            list.add(new String[]{prefix, Integer.toString(counter), "Random Forest", Double.toString(eval.precision(1)),
                    Double.toString(eval.recall(1)), Double.toString(eval.areaUnderROC(1)), Double.toString(eval.kappa())});
        }

        if (value == 1) {
            Instances training = train.getDataSet();

            int numAttr = training.numAttributes();
            training.setClassIndex(numAttr - 1);
            testing.setClassIndex(numAttr - 1);

            classifier.buildClassifier(training);
            Evaluation eval = new Evaluation(testing);

            eval.evaluateModel(classifier, testing);

            list.add(new String[]{prefix, Integer.toString(counter), "Random Forest", Double.toString(eval.precision(1)),
                    Double.toString(eval.recall(1)), Double.toString(eval.areaUnderROC(1)), Double.toString(eval.kappa())});
        }

    }

    private void calculateNaiveBayes(ConverterUtils.DataSource train, ConverterUtils.DataSource test, int value, int counter, List<String[]> list) throws Exception {
        Instances testing = test.getDataSet();
        NaiveBayes classifier = new NaiveBayes();

        if (value == 0) {

            int numAttr = testing.numAttributes();
            testing.setClassIndex(numAttr - 1);


            classifier.buildClassifier(testing);

            Evaluation eval = new Evaluation(testing);

            eval.evaluateModel(classifier, testing);

            list.add(new String[]{prefix, Integer.toString(counter), "Naive Bayes", Double.toString(eval.precision(1)),
                    Double.toString(eval.recall(1)), Double.toString(eval.areaUnderROC(1)), Double.toString(eval.kappa())});
        }

        if (value == 1) {
            Instances training = train.getDataSet();

            int numAttr = training.numAttributes();
            training.setClassIndex(numAttr - 1);
            testing.setClassIndex(numAttr - 1);


            classifier.buildClassifier(training);


            Evaluation eval = new Evaluation(testing);

            eval.evaluateModel(classifier, testing);

            list.add(new String[]{prefix, Integer.toString(counter), "Naive Bayes", Double.toString(eval.precision(1)),
                    Double.toString(eval.recall(1)), Double.toString(eval.areaUnderROC(1)), Double.toString(eval.kappa())});
        }

    }

    private void calculateIBK(ConverterUtils.DataSource train, ConverterUtils.DataSource test, int value, int counter, List<String[]> list) throws Exception {
        Instances testing = test.getDataSet();
        IBk classifier = new IBk();

        if (value == 0) {


            int numAttr = testing.numAttributes();

            testing.setClassIndex(numAttr - 1);


            classifier.buildClassifier(testing);

            Evaluation eval = new Evaluation(testing);

            eval.evaluateModel(classifier, testing);


            list.add(new String[]{prefix, Integer.toString(counter), "IBK", Double.toString(eval.precision(1)),
                    Double.toString(eval.recall(1)), Double.toString(eval.areaUnderROC(1)), Double.toString(eval.kappa())});
        }

        if (value == 1) {
            Instances training = train.getDataSet();

            int numAttr = training.numAttributes();
            training.setClassIndex(numAttr - 1);
            testing.setClassIndex(numAttr - 1);

            classifier.buildClassifier(training);

            Evaluation eval = new Evaluation(testing);

            eval.evaluateModel(classifier, testing);

            list.add(new String[]{prefix, Integer.toString(counter), "IBK", Double.toString(eval.precision(1)),
                    Double.toString(eval.recall(1)), Double.toString(eval.areaUnderROC(1)), Double.toString(eval.kappa())});
        }

    }

    private void writeCSV(List<String[]> wrt) {

        try (FileWriter fileWriter = new FileWriter(out);
             CSVWriter csvWriter = new CSVWriter(fileWriter)) {

            csvWriter.writeAll(wrt);
            csvWriter.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws Exception {

        importResources(1);
        new WekaEngine().writeCSV(new WekaEngine().walkForwardValidation(Integer.parseInt(numRelease)));
    }
}
