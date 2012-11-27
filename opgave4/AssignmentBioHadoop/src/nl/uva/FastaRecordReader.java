package nl.uva;

import java.io.IOException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.LineReader;

/**
 *
 * @author S. koulouzis
 */
public class FastaRecordReader implements org.apache.hadoop.mapred.RecordReader<LongWritable, Text> {

    private final long start;
    private final long length;
    private final long end;
    private final JobConf jc;
    private final CompressionCodec codec;
    private final FSDataInputStream fileIn;
    private Text value;
    private LongWritable key;
    private long pos;
    
    Log log = LogFactory.getLog(FastaInputFormat.class);

    public FastaRecordReader(InputSplit is, JobConf jc) throws IOException {
        FileSplit split = (FileSplit) is;
        start = split.getStart();
        length = split.getLength();
        end = start + length;

        this.jc = jc;
        CompressionCodecFactory compressionCodecs = new CompressionCodecFactory(jc);
        codec = compressionCodecs.getCodec(split.getPath());

        FileSystem fs = split.getPath().getFileSystem(jc);
        fileIn = fs.open(split.getPath());
        this.pos = start;
    }

    @Override
    public boolean next(LongWritable k, Text v) throws IOException {
        //Every time this method returns true we call a new map
        
        //If we read the entire chunk return false 
        if (pos >= end) {
            return false;
        }
        
        //Point to the correct position of the file
        fileIn.seek(pos);
        LineReader lr;
        //Create a line reader to read lines from the FASTA file 
        if (codec != null) {
            lr = new LineReader(codec.createInputStream(fileIn), jc);
        } else {
            lr = new LineReader(fileIn, jc);
        }

        int num = 0;
        Text line = new Text();
        StringBuilder sequence = new StringBuilder();
        int recordsRead = 0;
        while (pos <= end) {
            num = lr.readLine(line);
            pos += num;
            if (line.toString().indexOf(">") >= 0 || pos >= end) {
                //Use key.set(pos);and value.set(sequence.toString()); to set the 
                // key value pairs for the mapper
            } else {
               //Use the sequence.append(line.toString()); to build the protein sequence
            }
        }
        return false;
    }

    @Override
    public LongWritable createKey() {
        if (key == null) {
            key = new LongWritable();
        }
        key.set(pos);
        return key;
    }

    @Override
    public Text createValue() {
        if (value == null) {
            value = new Text();
        }
        return value;
    }

    @Override
    public long getPos() throws IOException {
        return pos;
    }

    @Override
    public void close() throws IOException {
        fileIn.close();
    }

    @Override
    public float getProgress() throws IOException {
        if (start == end) {
            return 0.0f;
        } else {
            return Math.min(1.0f, (pos - start) / (float) (end - start));
        }
    }
}
