package milestone.due.writer;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

public class PropertiesWriter {

    public static void main(String[] args){

        final Logger logger = Logger.getLogger(PropertiesWriter.class.getName());

        try (OutputStream output = new FileOutputStream("C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\config.properties")) {

            Properties prop = new Properties();

            // set the properties value for BOOKKEEPER Project
            prop.setProperty("M1D2BOOK", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\csvFile\\M1D2BOOK.csv");
            prop.setProperty("M1D2TAJO", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\csvFile\\M1D2TAJO.csv");

            prop.setProperty("M1D2BOOKARFF", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\arffFile\\M1D2BOOK.arff");
            prop.setProperty("M1D2TAJOARFF", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\arffFile\\M1D2BOOK.arff");

            prop.setProperty("prefixBOOK", "BOOK");
            prop.setProperty("prefixTAJO", "TAJO");

            prop.setProperty("M1D2TESTBOOK", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outCSVmethods\\testing");
            prop.setProperty("M1D2TRAINBOOK", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outCSVmethods\\training");

            prop.setProperty("M1D2TESTTAJO", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outCSVmethods\\testing");
            prop.setProperty("M1D2TRAINTAJO", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outCSVmethods\\training");

            prop.setProperty("OUTBOOK", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outMilestone2\\outM1D2BOOK.csv");
            prop.setProperty("OUTTAJO", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\outMilestone2\\outM1D2TAJO.csv");

            prop.setProperty("NUMBOOK", "7");
            prop.setProperty("NUMTAJO", "5");

            prop.setProperty("BOOKARFFTESTING", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\arffFile\\testing");
            prop.setProperty("TAJOARFFTESTING", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\arffFile\\testing");

            prop.setProperty("BOOKARFFTRAINING", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\arffFile\\training");
            prop.setProperty("TAJOARFFTRAINING", "C:\\Users\\Alessio Mazzola\\IdeaProjects\\Dev2M2\\src\\main\\resources\\arffFile\\training");






            // save properties to project root folder
            prop.store(output, null);

            String properties = String.valueOf(prop);

            logger.log(Level.INFO, properties);

        } catch (IOException e) {
            logger.log(Level.WARNING, String.valueOf(e));
        }
    }
}
