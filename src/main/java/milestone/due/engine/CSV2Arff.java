package milestone.due.engine;


//import required classes

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;


public class CSV2Arff {


    private static final Logger LOGGER = Logger.getLogger(CSV2Arff.class.getName());

    static String csvPathTraining = "";
    static String csvPathTesting = "";

    static String arffPathTraining = "";
    static String arffPathTesting = "";

    static String prefix = "";
    static String numRelease = "";

    static final String PRF_NAME = "\\M1D2";


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
                csvPathTraining = prop.getProperty("M1D2TRAINBOOK");
                csvPathTesting = prop.getProperty("M1D2TESTBOOK");

                arffPathTraining = prop.getProperty("BOOKARFFTRAINING");
                arffPathTesting = prop.getProperty("BOOKARFFTESTING");

                prefix = prop.getProperty("prefixBOOK");
                numRelease = prop.getProperty("NUMBOOK");


            } else {
                csvPathTraining = prop.getProperty("M1D2TRAINTAJO");
                csvPathTesting = prop.getProperty("M1D2TESTTAJO");

                arffPathTraining = prop.getProperty("TAJOARFFTRAINING");
                arffPathTesting = prop.getProperty("TAJOARFFTESTING");

                prefix = prop.getProperty("prefixTAJO");
                numRelease = prop.getProperty("NUMTAJO");


            }

        } catch (IOException e) {
            LOGGER.log(Level.WARNING, String.valueOf(e));
        }
    }

    private void makeArffTest(int value) throws IOException {

        if (value == 0) { //--> 0 testing
            for (int i = 1; i <= Integer.parseInt(numRelease); i++) { //testing

                // load CSV
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(csvPathTesting + PRF_NAME + prefix + i + "testing.csv"));
                Instances data = loader.getDataSet();//get instances object

                // save ARFF
                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);//set the dataset we want to convert
                //and save as ARFF
                saver.setFile(new File(arffPathTesting + PRF_NAME + prefix + i + "testing.arff"));
                saver.writeBatch();

            }
        } else if (value == 1) { // --> 1 training
            for (int i = 2; i <= Integer.parseInt(numRelease); i++) { //testing

                // load CSV
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(csvPathTraining + PRF_NAME + prefix + i + "training.csv"));
                Instances data = loader.getDataSet();//get instances object

                // save ARFF
                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);//set the dataset we want to convert
                //and save as ARFF
                saver.setFile(new File(arffPathTraining + PRF_NAME + prefix + i + "training.arff"));
                saver.writeBatch();
            }
        }

    }


    public static void main(String[] args) throws Exception {

        importResources(1);
        new CSV2Arff().makeArffTest(1);


    }
}
