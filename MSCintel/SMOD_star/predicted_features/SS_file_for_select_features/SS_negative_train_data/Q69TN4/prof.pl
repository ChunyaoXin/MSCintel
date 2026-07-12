#!/usr/bin/perl -w
use strict;

`hostname`=~/(\S+)/;
my $node=$1;
my $datetime=`date`;
my $path=`pwd`;

print "Hostname: $node\n";
print "Started : $datetime\n";
print "Path    : $path\n";


#======BEGIN======
my $datadir     = "/home/gongxiaotong/data1/negativetest/Q69TN4";
my $name        = "Q69TN4";
my $user        = "gongxiaotong";
my $tag         = "Q69TN4_prof";


#$datadir     = "/home/yangjy/project/peptide/Dataset/1a81A";
#$name        = "1a81A";
#$user        = "yangjy";
#$tag         = "$name\_spx";

if(!-s "$datadir/seq_fasta.fasta")
{
    die "$datadir/seq_fasta.fasta missed\n";
}


my $workdir  = "/tmp/$user/$tag";
`mkdir -p $workdir`;
# `rm -rf $workdir/*`;
chdir "$workdir";


`cp $datadir/seq_fasta.fasta .`;
my $spxdir="/home/luoyx/software/SPARKS-X/sparks-x";
#my $hhdir="/home/yangjy/software/profile";

`$spxdir/bin/psiblast.sh seq_fasta.fasta`;
#`$hhdir/run_hhblits seq.fasta`;
`cp $datadir/seq_fasta.fasta.pssm .`;
`$spxdir/SPINE-X/spineX.pl seq_fasta.fasta.pssm`;

`cp seq_fasta.fasta.phipsi $datadir/seq.phipsi`;
`cp seq_fasta.fasta.pssm $datadir/seq.pssm`;
#`cp seq.hhm $datadir/seq.hhm`;


#======END======
my $time=`date`;
printf "ending time: $time";
`sync`;
`sync`;
sleep(1);
# `rm -fr $workdir`;

exit();


