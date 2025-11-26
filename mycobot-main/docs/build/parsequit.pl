#!/usr/bin/perl

while (<>)
 {
  if ( $_ =~ /\@quitf/ )
   {
    $line = <>;
    print $line;
    exit;
    while ( ($line !~ /{/) & <>)
     {
      print $line; 
      $line = <>;
     }
    exit;
   }
  elsif ( $_ =~ /\@classf/ )
   {
    chomp($_);
    $_=~m/\@classf[ ]*([^\Z]*)\Z/;
    $namespace = $1;

    $line = <>;
    while ($line !~ /class/)
     {
      print "$line";
      $line = <>;
     }

    print $line
    $line=~m/class[ ]*([^\Z]*)\Z/;
    $restOfLine = $1;
    print "class $namespace\:\:$restOfLine";
   }
  elsif ( $_ =~ /\#\</ )
   {
    my($before, $after) = split(/\#\</, $_, 2);
    my($wspaces) = $before =~ /(\s+)/;
    print "$wspaces## $after";
    print "$before\n";
   }
  else
   {
    print;
   }
 }
#while (<>)
# {
#  if ($_ =~ /{/)
#   {
#    last;
#   }
#  else
#   {
#    print; 
#   }
##  else 
##   {
##    quit;
##   }
# }

