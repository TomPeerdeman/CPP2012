package nl.uva;

import java.io.*;
import java.util.*;
import org.biojava.bio.BioException;
import org.biojava.bio.alignment.AlignmentAlgorithm;
import org.biojava.bio.alignment.AlignmentPair;
import org.biojava.bio.seq.Sequence;
import org.biojava3.alignment.Alignments;
import org.biojava3.alignment.template.Profile;
import org.biojava3.core.sequence.ProteinSequence;
import org.biojava3.core.sequence.compound.AminoAcidCompound;
import org.biojava3.core.util.ConcurrencyTools;
import org.biojavax.SimpleNamespace;
import org.biojavax.bio.seq.RichSequence;
import org.biojavax.bio.seq.RichSequenceIterator;

/**
 *
 * @author S. Koulouzis
 */
public class AssignmentSequential {

    public static void main(String[] args) {
        if (args == null || args.length < 4 || args[0].equals("-help") || args[0].equals("-help")) {
            printHelp();
            System.exit(-1);
        }
        try {
            String queryProteinFile = args[0]; //"data/query.fa";
            String datasetFile = args[1]; //"data/Homo_sapiens.GRCh37.69.pep.abinitio.fa";
            String substitutionMatrixFile = args[2]; //"data/BLOSUM100"; 
            int numberOfrelative = Integer.valueOf(args[3]);

            //Load query fasta file
            RichSequenceIterator queryIter = getSequenceIterator(queryProteinFile);
            //Obtain query sequence
            RichSequence query = queryIter.nextRichSequence();

            //Start the pairwise alignment
            System.out.println("Getting the " + numberOfrelative + " most similar sequences\n");
            Map<ProteinSequence, Integer> topSequences = doPairwiseAlignment(query, datasetFile, substitutionMatrixFile, numberOfrelative);
            Set<ProteinSequence> keys = topSequences.keySet();
            
            System.out.println("Score\tSequence");
            System.out.println("__________________________");
            for(ProteinSequence s : keys){
                System.out.println(topSequences.get(s)+"\t"+s);
            }
            System.out.println();
            
            //Add the query sequence so we can do the multialignment
            topSequences.put(new ProteinSequence(query.seqString()), Integer.MAX_VALUE);
            //Sort the results so we can align from highest to lowest score
            Map<ProteinSequence, Integer> sortedSequences = sortSequencesByValue(topSequences);

            //Start the multialignment
            System.out.println("Performnig multialignment");
            System.out.println("__________________________");
            Profile<ProteinSequence, AminoAcidCompound> profile = doMultiAlignment(sortedSequences);
            System.out.println(profile);


        } catch (Exception ex) {
            printHelp();
            ex.printStackTrace();
        }
    }

    private static void printHelp() {
        System.out.println("Usage: <query protein file> <dataset file> <substitution matrix file> <number Of relative proteins>\n");
        System.out.println("query protein file : 		The query protein fasta file. If this file contains more than one sequences the code will load only the first.\n"
                + "dataset file: 			The dataset fasta file containing the sequences we want to find the compare the query sequence with.\n"
                + "substitution matrix file:       Is the file containing the substitution matrix for initilizing Needleman Wunsch alignment algorithm.\n"
                + "number Of relative proteins :   The number of the higest scoring sequences we want to allign.\n");
    }

    private static RichSequenceIterator getSequenceIterator(String queryProteinFile) throws FileNotFoundException {
        //Create a reader 
        BufferedReader br = new BufferedReader(new FileReader(queryProteinFile));
        //Create a dummy namespace
        SimpleNamespace ns = new SimpleNamespace("biojava");
        //Get the iterator 
        return RichSequence.IOTools.readFastaProtein(br, ns);
    }

    private static Map<ProteinSequence, Integer> doPairwiseAlignment(RichSequence query, String datasetFile, String substitutionMatrixFile, int numberOfrelative) throws FileNotFoundException, NoSuchElementException, NumberFormatException, BioException, IOException, Exception {
        //Load the Substitution Matrix
        org.biojava.bio.alignment.SubstitutionMatrix matrix = new org.biojava.bio.alignment.SubstitutionMatrix(new File(substitutionMatrixFile));
        //Create the Needleman Wunsch Alignment Algorithm (see http://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
        AlignmentAlgorithm aligner = new org.biojava.bio.alignment.NeedlemanWunsch(
                (short) 0, // match
                (short) 3, // replace
                (short) 2, // insert
                (short) 2, // delete
                (short) 1, // gapExtend
                matrix // SubstitutionMatrix
                );

        //Create the array that will hold the top scores 
        int[] maxScores = new int[numberOfrelative];
        //Create the array that will hold the top sequences
        ProteinSequence[] topSequences = new ProteinSequence[numberOfrelative];
        String[] topSequencesNames = new String[numberOfrelative];
        for (int i = 0; i < numberOfrelative; i++) {
            maxScores[i] = Integer.MIN_VALUE;
        }
        //Load the dataset 
        RichSequenceIterator seqIter = getSequenceIterator(datasetFile);


        //Here we iterate through all the sequences and getting the score with the query sequence.
        //We use the maxScores array to find the N highest scores. 
        //For every score we compare the the values in the maxScores array. If 
        //the score is larger than the value at index hold index.
        //Get the value at index and see if we are going to shift it 
        while (seqIter.hasNext()) {
            int index = -1;
            Sequence target = seqIter.nextSequence();
            //Get the score 
            AlignmentPair pair = aligner.pairwiseAlignment(query, target);
            int score = pair.getScore();
            //Find if this score is larger than any existing 
            for (int i = 0; i < numberOfrelative; i++) {
                if (score > maxScores[i]) {
                    //Keep the index 
                    index = i;
                }
            }
            if (index != -1) {
                //Get the previous score
                Integer prevScore = maxScores[index];
                //Get the previous sequence
                ProteinSequence prevSqu = topSequences[index];
                if (index > 0) {
                    //If the previous score is larger than the value before it replace it 
                    if (prevScore > maxScores[index - 1]) {
                        maxScores[index - 1] = prevScore;
                        topSequences[index - 1] = prevSqu;
                    }
                }
                maxScores[index] = score;
                //Add the sequence
                topSequences[index] = new ProteinSequence(target.seqString());
                topSequencesNames[index] = target.getName();
            }
        }
        Map<ProteinSequence, Integer> map = new HashMap<ProteinSequence, Integer>();
        //Add them in opposite order
        for (int i = 0; i < maxScores.length; i++) {
            map.put(topSequences[i], maxScores[i]);
        }

        return map;
    }

    private static Profile<ProteinSequence, AminoAcidCompound> doMultiAlignment(Map<ProteinSequence, Integer> topSequences) {
        List<ProteinSequence> list = new ArrayList<ProteinSequence>(topSequences.keySet());
        Profile<ProteinSequence, AminoAcidCompound> profile = Alignments.getMultipleSequenceAlignment(list);
        ConcurrencyTools.shutdown();
        return profile;
    }

    private static Map<ProteinSequence, Integer> sortSequencesByValue(Map<ProteinSequence, Integer> topSequences) {
        ValueComparator vc = new ValueComparator(topSequences);
        TreeMap<ProteinSequence, Integer> sorted_map = new TreeMap<ProteinSequence, Integer>(vc);
        sorted_map.putAll(topSequences);
        return sorted_map;
    }
}
