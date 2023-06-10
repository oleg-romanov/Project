using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;


public class GameSelector : MonoBehaviour
{
    public void StartFlowerGame()
    {
        SceneManager.LoadScene("FlowerGame");
    }

    public void StartCrossesGame()
    {
        SceneManager.LoadScene("CrossesGame");
    }

    public void StartLeftRightGame()
    {
        SceneManager.LoadScene("SideToSideMovements");
    }

    public void StartTopDownGame()
    {
        SceneManager.LoadScene("TopAndBottomMovements");
    }

    public void StartFocusingGame()
    {
        SceneManager.LoadScene("SampleScene");
    }
}
