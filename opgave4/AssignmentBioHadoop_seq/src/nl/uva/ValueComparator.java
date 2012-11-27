/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.uva;

import java.util.Comparator;
import java.util.Map;
import org.biojava3.core.sequence.ProteinSequence;

/**
 *
 * @author S. Koulouzis
 */
class ValueComparator implements Comparator<ProteinSequence> {

    Map<ProteinSequence, Integer> base;

    public ValueComparator(Map<ProteinSequence, Integer> base) {
        this.base = base;
    }

    @Override
    public int compare(ProteinSequence a, ProteinSequence b) {
        if (base.get(a) >= base.get(b)) {
            return -1;
        } else {
            return 1;
        } // returning 0 would merge keys
    }
}
