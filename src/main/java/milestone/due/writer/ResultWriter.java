package milestone.due.writer;

import com.opencsv.CSVWriter;
import milestone.due.engine.MakeARFFFile;
import milestone.due.engine.WekaEngine;

import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ResultWriter {

    /**
     * Questa classe ha il compito di andare a creare il file CSV risultate dalle misurazioni
     * In particolare le colonne presenti nel CSV saranno:
     *
     * [DATASET, #TRAININGRELEASE, CLASSIFIER, PRECISION, RECALL, AUC, KAPPA]
     */

    private static final Logger LOGGER = Logger.getLogger(ResultWriter.class.getName());

    static String m1d2Test = "";
    static String m1d2Train = "";
    static String prefix = "";

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
                m1d2Test = prop.getProperty("M1D2TESTBOOK");
                m1d2Train = prop.getProperty("M1D2TRAINBOOK");
                prefix = prop.getProperty("prefixBOOK");
            } else {
                m1d2Test = prop.getProperty("M1D2TESTTAJO");
                m1d2Train = prop.getProperty("M1D2TRAINTAJO");
                prefix = prop.getProperty("prefixTAJO");
            }

        } catch (IOException e) {
            LOGGER.log(Level.WARNING, String.valueOf(e));
        }
    }


    public static void main(String[] args) throws Exception {

        importResources(0);

        List<String[]> wrt;
        wrt = new WekaEngine().walkForwardValidation(7);

        try(FileWriter fileWriter = new FileWriter("C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outMilestone2\\out.csv");
            CSVWriter csvWriter = new CSVWriter(fileWriter)) {

            csvWriter.writeAll(wrt);
            csvWriter.flush();

        }


    }
}
