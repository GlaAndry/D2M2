package milestone.due.engine;

import com.opencsv.CSVReader;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MakeARFFFile {
    /**
     * Questa classe è necessaria al fine di trasformare un file CSV da analizzare
     * in un file ARFF, così da sfruttare al meglio WEKA
     * <p>
     * Per fare un file ARFF andiamo prima a leggere il file CSV da trasformare in modo da
     * estrapolare i parametri dalla prima riga.
     *
     * @relation name of CSV
     * @attributes List of attributes of CSV
     * .
     * .
     * .
     * @data Actual data of CSV
     */

    private static final Logger LOGGER = Logger.getLogger(MakeARFFFile.class.getName());

    static String m1d2Book = "";
    static String m1d2Tajo = "";
    static String m1d2BookArff = "";
    static String m1d2TajoArff = "";

    private static void importResources() {
        /**
         * Attraverso config.properties andiamo a caricare i valori delle stringhe per le open e le write dei file.
         * Necessario al fine di evitare copie inutili dello stesso codice in locazioni diverse della classe.
         */
        try (InputStream input = new FileInputStream("C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\config.properties")) {

            Properties prop = new Properties();
            // load a properties file
            prop.load(input);

            m1d2Book = prop.getProperty("M1D2BOOK");
            m1d2Tajo = prop.getProperty("M1D2TAJO");
            m1d2BookArff = prop.getProperty("M1D2BOOKARFF");
            m1d2TajoArff = prop.getProperty("M1D2TAJOARFF");


        } catch (IOException e) {
            LOGGER.log(Level.WARNING, String.valueOf(e));
        }
    }


    private void makeArff(CSVReader csvReader, String path, String csvName) throws IOException {

        Scanner scanner = new Scanner(System.in);
        String[] parameters;
        parameters = csvReader.readNext(); //prendo la prima riga del file CSV

        List<String[]> csvData = new ArrayList<>();
        csvData = csvReader.readAll();

        int counter=0;

        System.out.println("Creo il file arff...");

        try (FileWriter fileWriter = new FileWriter(path)) {
            fileWriter.append("@relation ").append(csvName);
            fileWriter.append("\n");

            for (int i = 0; i < parameters.length; i++) {
                System.out.println("attributo: "+ parameters[i]);
                fileWriter.append("@attribute ").append(parameters[i]).append(" ").append(scanner.nextLine());
                fileWriter.append("\n");
            }
            fileWriter.append("@data");
            fileWriter.append("\n");

            for (String[] strings : csvData) {
                for (int z = 0; z < parameters.length; z++) {
                    counter++;
                    fileWriter.append("'");
                    fileWriter.append(strings[z]);
                    if (counter == parameters.length) {
                        fileWriter.append("'");
                    } else {
                        fileWriter.append("',");
                    }

                }
                counter = 0;
                fileWriter.append("\n");

            }


        }

        System.out.println("Fatto!");

    }

    public static void main(String[] args) throws IOException {

        importResources();

        FileReader fileReader = new FileReader(m1d2Book);
        CSVReader csvReader = new CSVReader(fileReader);

        new MakeARFFFile().makeArff(csvReader, m1d2BookArff, "M1D2BOOK");

    }
}
