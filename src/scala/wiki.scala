package nlpproject

import java.io.File
import scala.collection.immutable.Map
import scala.collection.JavaConverters._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.conf.Configuration

import se.lth.cs.docria.Document
import se.lth.cs.docria.Layer
import se.lth.cs.docria.Node
import se.lth.cs.docria.hadoop.DocriaInputFormat
import se.lth.cs.docria.hadoop.DocumentWritable


object Wiki {
  def main(args: Array[String]) {
    //val ancs = for {
      //(k, v) <- rdd
      //anchors <- anchor(k.getDocument())
    //} yield anchors.size()

    //val count = ancs reduce((x,y) => x+y)
    val sparkConf = new SparkConf().setAppName("Wiki").setMaster("local")
    val sc = new SparkContext(sparkConf)
    //sc.setLogLevel("WARN")
    val conf = new Configuration()
    readDocria(sc, conf, args(0))
    sc.stop()
  }
  def readDocria(sc: SparkContext, conf: Configuration, f: String) {
    val path = new File(f).getCanonicalPath()
    val fname = s"file://$path"

    val rdd = sc.newAPIHadoopFile(fname, classOf[DocriaInputFormat],
       classOf[DocumentWritable], classOf[NullWritable], conf)

    val targets = rdd flatMap { case (k, v) => allNonSelfWikiLinks(k.getDocument()) }
    val count = targets groupBy(_._1) map {case (k, v) => Array(k, counts(v map(_._2))).mkString(",")}

    count.saveAsTextFile("wiki_mappings")
  }

  def anchor(doc: Document) : Option[Layer] = Option(doc.layer("anchor"))
  def nodeGet(node: Node, field: String) = node.get(field).stringValue()
  def internal(node: Node) : Boolean = nodeGet(node, "target_type") == "internal"

  def allNonSelfWikiLinks(doc: Document) : List[(String, Int)] = {
    val props = doc.props()
    if (props.containsKey("wkd")) {
      val docid : Int = props.get("wkd").intValue()
      anchor(doc).toList flatMap { l =>
        l.iterator().asScala filter(internal _) map(x => (nodeGet(x, "text"), x.get("target_wkd").intValue)) filter(_._2 != docid)
      } 
    }  else {
      List[(String, Int)]()
    }
  }
  def counts(itr : Iterable[Int]): Int = itr groupBy(x => x) maxBy {case (k,v) => v.size} _1
}
