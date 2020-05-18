package milestone.due.engine;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DivideCSVByRelease {

    /**
     * Questa classe prende in input il CSV della milestone precedente e lo divide in release, in modo tale
     * da poter utilizzare il metodo Walk Forward attraverso Weka.
     */

    private static final Logger LOGGER = Logger.getLogger(MakeARFFFile.class.getName());

    static String m1d2Path = "";
    static String prefix = "";
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
                m1d2Path = prop.getProperty("M1D2BOOK");
                prefix = prop.getProperty("prefixBOOK");
                numRelease = prop.getProperty("NUMBOOK");

            } else {
                m1d2Path = prop.getProperty("M1D2TAJO");
                prefix = prop.getProperty("prefixTAJO");
                numRelease = prop.getProperty("NUMTAJO");

            }

        } catch (IOException e) {
            LOGGER.log(Level.WARNING, String.valueOf(e));
        }
    }

    private void walkForwardDivisorTraining(CSVReader csvReader, int countRelease, FileWriter fileWriter) throws IOException {

        /**
         * La divisione attraverso il numero di release ci permette di utilizzare al meglio il metodo
         * walk forward
         */

        List<String[]> read = csvReader.readAll();
        List<String[]> read2 = read.subList(1, read.size()); // rimuovo la prima stringa.

        String release = read2.get(0)[0];
        String lastRelase = "";

        List<String[]> out = new ArrayList<>();

        out.add(new String[]{"Version", "Class", "NFix", "NRevision", "NAuthor", "LocAdded", "LocDeleted", "LocTouched", "MaxLockAdded", "AverageLockAdded", "Size", "Age", "Buggy"}); //header necessario
        //altrimenti weka genera errore.

        for (String[] str : read2) {
            if ((countRelease - 1) != 0) {
                lastRelase = str[0];
                if (release.equals(str[0])) {
                    out.add(new String[]{str[0], str[1], str[2], str[3], str[4], str[5], str[6], str[7], str[8], str[9], str[10], str[11], str[12]});
                } else {
                    countRelease--;
                    release = str[0];
                    out.add(new String[]{str[0], str[1], str[2], str[3], str[4], str[5], str[6], str[7], str[8], str[9], str[10], str[11], str[12]});
                }
            }

        }

        if (lastRelase.equals(release)) {
            out.remove(out.size() - 1); //rimuoviamo la riga iniziale della release successiva, se esiste.
        }

        try (CSVWriter csvWriter = new CSVWriter(fileWriter)) {

            csvWriter.writeAll(out);
            csvWriter.flush();

        }

    }

    private void walkForwardDivisorTesting(CSVReader csvReader, int countRelease, FileWriter fileWriter) throws IOException {

        /**
         * La divisione attraverso il numero di release ci permette di utilizzare al meglio il metodo
         * walk forward
         */

        List<String[]> read = csvReader.readAll();
        List<String[]> read2 = read.subList(1, read.size()); // rimuovo la prima stringa.

        String release = read2.get(0)[0];
        String lastRelase = "";

        List<String[]> out = new ArrayList<>();

        out.add(new String[]{"Version", "Class", "NFix", "NRevision", "NAuthor", "LocAdded", "LocDeleted", "LocTouched", "MaxLockAdded", "AverageLockAdded", "Size", "Age", "Buggy"}); //header necessario
        //altrimenti weka genera errore.

        for (String[] str : read2) {
            if (Integer.toString(countRelease).equals(str[0])) {
                out.add(new String[]{str[0], str[1], str[2], str[3], str[4], str[5], str[6], str[7], str[8], str[9], str[10], str[11], str[12]});
            }
        }


        try (CSVWriter csvWriter = new CSVWriter(fileWriter)) {

            csvWriter.writeAll(out);
            csvWriter.flush();

        }

    }

    private void divisorTraining(int count) {
        /**
         * Questo metodo va a creare tutti i file CSV necessari per eseguire il walk forward.
         * Il ciclo parte da i = 2 in quanto per il primo run si ha solamente testing senza training
         */

        for (int i = 2; i <= count; i++) {
            try (FileReader fileReader = new FileReader(m1d2Path);
                 CSVReader csvReader = new CSVReader(fileReader);
                 FileWriter fileWriter = new FileWriter("C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outCSVmethods\\training\\M1D2" + prefix + i + "training.csv")) {

                new DivideCSVByRelease().walkForwardDivisorTraining(csvReader, i, fileWriter);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }


    }

    private void divisorTesting(int count) {


        for (int i = 1; i <= count; i++) {
            try (FileReader fileReader = new FileReader(m1d2Path);
                 CSVReader csvReader = new CSVReader(fileReader);
                 FileWriter fileWriter = new FileWriter("C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outCSVmethods\\testing\\M1D2" + prefix + i + "testing.csv")) {

                new DivideCSVByRelease().walkForwardDivisorTesting(csvReader, i, fileWriter);


            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }


    public static void main(String[] args) {

        importResources(1);
        new DivideCSVByRelease().divisorTraining(Integer.parseInt(numRelease));
        new DivideCSVByRelease().divisorTesting(Integer.parseInt(numRelease));

    }
}
