using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChessCell : MonoBehaviour
{
    public int rowIndex;
    public int columnIndex;
    public bool isBlack;
    public bool isTarget;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void configure(int rowIndex, int columnIndex, bool isBlack, bool isTarget)
    {
        this.rowIndex = rowIndex;
        this.columnIndex = columnIndex;
        this.isBlack = isBlack;
        this.isTarget = isTarget;
    }

    private void OnMouseEnter()
    {
        if (isTarget)
        {
            ChessBoardScript.targetIsHit = true;
            Debug.Log("Выбрана ячейка: (" + rowIndex + "," + columnIndex + ")");
        }
        else
        {
            return;
        }
    }
}
