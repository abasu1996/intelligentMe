from fastapi import FastAPI
from pydantic import BaseModel
from lxml import etree
from typing import List, Dict
import uvicorn

app = FastAPI(title="Tax Agent API")

invoice_xml_example = """<Invoice currency="USD">
    <TaxAmount>15.00</TaxAmount>
    <line id="1" amount="100.00" tax="10.00"/>
    <line id="2" amount="50.00" tax="5.00"/>
</Invoice>"""

po_xml_example = """<PurchaseOrder currency="USD">
    <TaxAmount>12.00</TaxAmount>
    <line id="1" amount="100.00" tax="8.00"/>
    <line id="2" amount="50.00" tax="4.00"/>
</PurchaseOrder>"""

class CompareRequest(BaseModel):
    invoice_xml:str
    po_xml:str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


print("Starting XML parsing utilities")
def parse_document(xml_string:str) -> Dict:
    print(xml_string)
    root = etree.fromstring(xml_string.encode("utf-8"))
    document = {
        "lines":[],
        "total_tax":0.0,
        "currency":None

    }

    tax_node = root.find(".//TaxAmount")
    if tax_node is not None:
        document["total_tax"] = float(tax_node.text)

    currency_node = root.find(".//Currency")

    if currency_node is not None:
        document["currency"] = currency_node.text
    
    for line in root.xpath('.//line'):
        document["lines"].append({
            "id": line.get("id"),
            "amount": float(line.get("amount", 0)),
            "tax": float(line.get("tax", 0))
        })
        document["total_tax"] += float(line.get("tax", 0))
    document["currency"] = root.get("currency")
    print(document)
    return document



def analyze_po_variance(po:Dict, invoice:Dict) -> Dict:
    report = {
        "header_variance":0,
        "line_variances":[],
        "recommended_actions":[]
    }

    header_diff = round(invoice["total_tax"] - po["total_tax"], 2)
    report["header_variance"] = header_diff

    if header_diff !=0:
        report["recommended_actions"].append("Header tax variance detected. Review total tax amounts.")
    elif header_diff == 0:
        report["recommended_actions"].append("No header tax variance detected. Proceed with line item analysis.")
    elif header_diff > 0:
        report["recommended_actions"].append("Invoice tax is higher than PO tax. Verify if additional charges are justified.")
    else:        report["recommended_actions"].append("Invoice tax is lower than PO tax. Check for missing tax lines or discounts.")

    for po_line in po["lines"]:
        invoice_line = next((line for line in invoice["lines"] if line["id"] == po_line["id"]), None)
        if invoice_line:
            line_diff = round(invoice_line["tax"] - po_line["tax"], 2)
            report["line_variances"].append({
                "line_id": po_line["id"],
                "tax_difference": line_diff
            })
        else:
            report["recommended_actions"].append(f"Line {po_line['id']} not found in invoice")

    return report

@app.post("/compare-tax")
def compare_tax(request: CompareRequest):
    print(" Received compare request")
    invoice_data = parse_document(request.invoice_xml)
    po_data = parse_document(request.po_xml)

    variance_report = analyze_po_variance(po_data, invoice_data)

    return {
        "status": "completed",
        "variance_analysis": variance_report
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
