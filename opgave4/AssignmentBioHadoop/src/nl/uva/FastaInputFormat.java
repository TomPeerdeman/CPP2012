package nl.uva;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.LineReader;

/**
 *
 * @author S. Koulouzis
 */
public class FastaInputFormat extends FileInputFormat<LongWritable, Text> {

    Log log = LogFactory.getLog(FastaInputFormat.class);
    private FileSystem fs;
    private int totalRecordsRead=0;

    @Override
    public RecordReader<LongWritable, Text> getRecordReader(InputSplit is, JobConf jc, Reporter rprtr) throws IOException {
        rprtr.setStatus(is.toString());
        return new FastaRecordReader(is, jc);
    }

    /**
     *
     * @param job
     * @param numSplits the desired number of splits, a hint.
     * @return
     */
    @Override
    public InputSplit[] getSplits(JobConf job, int numSplits) throws FileNotFoundException, IOException {
        fs = FileSystem.get(job);
        List<FileSplit> splits = new ArrayList<FileSplit>();
        Path[] inputPaths = FastaInputFormat.getInputPaths(job);
        
        //Get the input paths and start spliting the FASTA files 
        for (Path p : inputPaths) {
            if (!fs.isFile(p)) {
                FileStatus[] status = fs.listStatus(p);
                for (FileStatus s : status) {
                    Path path = s.getPath();
                    splits.addAll(getSplitsForFile(path, job, numSplits));
                }
            } else {
                splits.addAll(getSplitsForFile(p, job, numSplits));
            }
        }
        log.info("Number of splis: " + splits.size() + ". Number of records: " + totalRecordsRead);
        return splits.toArray(new org.apache.hadoop.mapred.InputSplit[splits.size()]);
    }

    public List<FileSplit> getSplitsForFile(Path inputPath, Configuration conf, int numLinesPerSplit) throws IOException {
        List<FileSplit> splits = new ArrayList<FileSplit>();
        FileSystem fs = inputPath.getFileSystem(conf);
        if (!fs.isFile(inputPath)) {
            throw new IOException("Not a file: " + inputPath);
        }
        LineReader lr = null;
        try {
            FSDataInputStream in = fs.open(inputPath);
            lr = new LineReader(in, conf);
            Text line = new Text();
            long start = 0;
            int recordsRead = 0;
            int bytesRead = 0;
            int totalbytesRead = 0;
            int num;
            //Get the number of records to add per split 
            Integer recordsPerSplit = conf.getInt("records.per.split", 1000);
            
            while ((num = lr.readLine(line)) > 0) {
                //We hit the starting line 
                if (line.toString().indexOf(">") >= 0 || totalbytesRead + num >= fs.getFileStatus(inputPath).getLen()) {
                    //Use splits.add(new FileSplit(inputPath, start, bytesRead, new String[]{}));
                    //to add splits. Where inputPath is the location of the FASTA 
                    //file in the DFS start is the start of the bytes to read and bytesRead the length 
                }
            }
            
            //Add the leftovers
            if (totalRecordsRead % recordsPerSplit != 0 || recordsPerSplit > totalRecordsRead) {
                splits.add(new FileSplit(inputPath, start, bytesRead, new String[]{}));
            }

        } finally {
            if (lr != null) {
                lr.close();
            }
        }
        return splits;
    }
}
