#include <stdio.h>
#include <libxml/parser.h>
#include <libxml/tree.h>


int main(int argc, char** argv)
{
    xmlDocPtr doc; //定义解析文件指针
    xmlNodePtr curNode; //定义结点指针，需要它在各个结点间移动
    xmlChar *szKey; //临时字符串变量

    char *szDocName;

    if (argc <= 1)
    {
        printf("Usage : %s docname \n", argv[0]);
        return -1;
    }

    szDocName = argv[1];

    /*解析文件*/
    doc = xmlParseFile(szDocName);

    if ( NULL == doc)
    {
        fprintf(stderr,"Document not parsed successfully. \n");
        return -1;
    }

    /*确定文件根元素*/
    curNode = xmlDocGetRootElement(doc);
    if ( NULL == curNode )
    {
        fprintf(stderr,"empty document \n"); 
        xmlFreeDoc(doc); 
        return -1; 
    }

    printf("curNode->name = %s \n", curNode->name);
    if ( xmlStrcmp( curNode->name, BAD_CAST"xmlinfo" ) )
    {
        fprintf(stderr,"document of the wrong type, root node != xmlinfo\n"); 
        xmlFreeDoc(doc); 
        return -1; 
    }

    curNode = curNode->children;
    xmlNodePtr propNodePtr = curNode;

    while ( curNode != NULL )
    {
        /*取出结点中的内容*/
        if ( !xmlStrcmp( curNode->name, BAD_CAST"version" ) )
        {
            szKey = xmlNodeGetContent(curNode);
            printf("version: %s \n", szKey); 
            xmlFree(szKey); 
        }

        /*取出结点中的内容*/
        if ( !xmlStrcmp( curNode->name, BAD_CAST"keyinfo" ) )
        {
            szKey = xmlNodeGetContent(curNode);
            printf("keyinfo: %s \n", szKey); 
            xmlFree(szKey); 
        }

        /*取出结点中的内容*/
        if ( !xmlStrcmp( curNode->name, BAD_CAST"chain" ) )
        {
            propNodePtr = curNode->children;
            while( propNodePtr != NULL )
            {
                /*取出结点中的内容*/
                if ( !xmlStrcmp( propNodePtr->name, BAD_CAST"body" ) )
                {
                    szKey = xmlNodeGetContent(propNodePtr);
                    printf("body: %s \n", szKey); 
                    xmlFree(szKey); 
                }

                propNodePtr = propNodePtr->next;
            }
        }

        curNode = curNode->next; 
    }

    xmlFreeDoc( doc );
    return 0;   
}


