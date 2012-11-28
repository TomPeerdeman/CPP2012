package nl.uva;

import java.io.*;
import java.net.URI;
import java.util.NoSuchElementException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.biojava.bio.BioException;
import org.biojava.bio.alignment.AlignmentAlgorithm;
import org.biojava.bio.alignment.AlignmentPair;
import org.biojava.bio.alignment.SubstitutionMatrix;
import org.biojava.bio.seq.Sequence;
import org.biojava.bio.seq.ProteinTools;
import org.biojavax.SimpleNamespace;
import org.biojavax.bio.seq.RichSequence;
import org.biojavax.bio.seq.RichSequenceIterator;

public class Map extends MapReduceBase implements Mapper<LongWritable, Text, ScoreWritable, Text> {

    private RichSequence query;
    private AlignmentAlgorithm aligner;
    Log log = LogFactory.getLog(nl.uva.Map.class);

    static enum Counters {

        INPUT_SEQUENCES
    }

    @Override
    public void configure(JobConf job) {
        try {
            //Before the Mapper starts we need to get the query sequence and 
            //the aligner
            URI[] inputFiles = DistributedCache.getCacheFiles(job);
            Path targetProteinFile = new Path(inputFiles[0]);
            
            Path substitutionMatrixFile = new Path(inputFiles[1]);
            
            RichSequenceIterator queryIter = getSequenceIterator(targetProteinFile, job);
            query = queryIter.nextRichSequence();

            FileSystem fs = FileSystem.get(substitutionMatrixFile.toUri(), job);
            File temp_data_file = File.createTempFile(substitutionMatrixFile.getName(), "");
            temp_data_file.deleteOnExit();
            fs.copyToLocalFile(substitutionMatrixFile, new Path(temp_data_file.getAbsolutePath()));


            aligner = getNeedlemanWunschAligner(temp_data_file);

        } catch (NoSuchElementException ex) {
            Logger.getLogger(AssignmentMapreduce.class.getName()).log(Level.SEVERE, null, ex);
        } catch (BioException ex) {
            Logger.getLogger(AssignmentMapreduce.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(AssignmentMapreduce.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void map(LongWritable key, Text value, OutputCollector<ScoreWritable, Text> oc, Reporter rprtr) throws IOException {
        try {
            //Each Mapper receives a sequence from the dataset and performs 
            //an Alignment.            
            
            //Here you need to create the target protein. Make sure you split the 
            //header from the sequence
            
            Sequence target = ProteinTools.createProteinSequence(value.toString(), "prot_" + key.get());
            //Use the Sequence target = Protein Tools.createProteinSequence(protein Sequence, name);
            
           
            AlignmentPair pair = aligner.pairwiseAlignment(query, target);
            //Get the score and emit it to the Reducers 
            int score = pair.getScore();
            
            //You have to emit the results to the reducer. You have to add the 
            //score as the key ecause mapper emmits the results in a sorted order
            //Use oc.collect(SCORE, TARGET_SEQUENCE); method to do that.
			oc.collect(new ScoreWritable(score), new Text(value));
            //Hint: to get the sequence as a string use target.seqString()
            
            //Report the progress
            rprtr.incrCounter(nl.uva.Map.Counters.INPUT_SEQUENCES, 1);
        } catch (Exception ex) {
            Logger.getLogger(AssignmentMapreduce.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private RichSequenceIterator getSequenceIterator(Path targetProteinFile, Configuration jobConf) throws FileNotFoundException, IOException {
        FileSystem fs = FileSystem.get(jobConf);
        DataInputStream d = new DataInputStream(fs.open(targetProteinFile));
        BufferedReader br = new BufferedReader(new InputStreamReader(d));
        SimpleNamespace ns = new SimpleNamespace("biojava");
        return RichSequence.IOTools.readFastaProtein(br, ns);
    }

    private AlignmentAlgorithm getNeedlemanWunschAligner(File substitutionMatrixFile) throws NumberFormatException, NoSuchElementException, BioException, IOException {
        //Get the substitution matrix. For more information on substitution 
        //matrix http://en.wikipedia.org/wiki/Substitution_matrix
        SubstitutionMatrix matrix = new SubstitutionMatrix(substitutionMatrixFile);
        return new org.biojava.bio.alignment.NeedlemanWunsch(
                (short) 0, // match
                (short) 3, // replace
                (short) 2, // insert
                (short) 2, // delete
                (short) 1, // gapExtend
                matrix // SubstitutionMatrix
                );
    }
}
