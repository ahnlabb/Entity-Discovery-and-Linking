module Main exposing (main)

import Browser
import Browser.Navigation as Navigation
import Dict
import Element exposing (Element, alignRight, alignTop, centerX, centerY, column, el, fill, height, none, padding, px, rgb255, row, spacing, width)
import Element.Background as Background
import Element.Border as Border
import Element.Events as Events
import Element.Font as Font
import Element.Input as Input
import Html exposing (option, select)
import Html.Attributes as HAttr exposing (class, style)
import Html.Events exposing (onInput)
import Http
import Json.Decode as Decode exposing (Decoder, dict, field, int, list, string)
import Json.Decode.Pipeline exposing (custom, hardcoded, required)
import Json.Encode as Encode
import Set
import Svg
import Svg.Attributes as SAttr
import Time
import Url exposing (Url)
import Url.Builder as UrlBuilder


main =
    Browser.application
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        , onUrlRequest = RequestedUrl
        , onUrlChange = ChangedUrl
        }



-- MODEL


type Model
    = Loading
    | Error Http.Error
    | NoModels
    | Done Page
    | News ApiData Page
    | LoadingNews


type alias ApiData =
    { query : String
    , index : Int
    , key : String
    }


type alias Page =
    { selection : String
    , models : List String
    , reduceTags : Bool
    , prediction : Maybe Document
    , text : String
    , links : List String
    , showingInfo : Set.Set Int
    }


emptyPage : String -> List String -> Page
emptyPage selection models =
    Page selection models True (Just emptyDocument) "" [] Set.empty


initNews selection models =
    News initApiData (emptyPage selection models)


initApiData =
    { query = "China", index = 0, key = "8e7937f06f5d43fa9f51f2d08258864f" }


nextResult apiData =
    { apiData | index = apiData.index + 1 }


type alias Document =
    { text : String
    , entities : List Tag
    , reduced : List Entity
    }


showInfo index page =
    { page | showingInfo = Set.insert index page.showingInfo }


hideInfo index page =
    { page | showingInfo = Set.remove index page.showingInfo }


emptyDocument : Document
emptyDocument =
    Document "" [] []


initEntity tag wikidata =
    Entity tag wikidata


type alias Tag =
    { start : Int
    , stop : Int
    , class : String
    }


type alias Entity =
    { tag : Tag
    , wikidata : Maybe Wikidata
    }


type alias Wikidata =
    { entity : String
    , image : String
    , name : String
    , link : String
    , description : String
    }


init : () -> Url -> Navigation.Key -> ( Model, Cmd Msg )
init _ { fragment } _ =
    case fragment of
        Just "news" ->
            ( LoadingNews, getModels )

        _ ->
            ( Loading, getModels )



-- UPDATE


type Msg
    = NewModels (Result Http.Error (List String))
    | NewNews (Result Http.Error String)
    | NewPrediction (Result Http.Error Document)
    | NewLinks (Result Http.Error (List String))
    | StartHoveringEntity Int
    | StopHoveringEntity Int
    | ToggleReduce Bool
    | NewSelection String
    | RequestedUrl Browser.UrlRequest
    | ChangedUrl Url
    | MoreNews


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case ( msg, model ) of
        ( NewModels result, Loading ) ->
            result
                |> handleResult
                    (\models ->
                        case models of
                            selection :: rest ->
                                ( Done (emptyPage selection rest), Cmd.none )

                            [] ->
                                ( NoModels, Cmd.none )
                    )

        ( NewModels result, LoadingNews ) ->
            result
                |> handleResult
                    (\models ->
                        case models of
                            selection :: rest ->
                                ( initNews selection rest, getNews initApiData )

                            [] ->
                                ( NoModels, Cmd.none )
                    )

        ( NewPrediction result, Done page ) ->
            updatePrediction Done page result

        ( NewPrediction result, News apiData page ) ->
            updatePrediction (News apiData) page result

        ( NewLinks result, Done page ) ->
            updateLinks Done page result

        ( NewLinks result, News apiData page ) ->
            updateLinks (News apiData) page result

        ( StartHoveringEntity index, News apiData page ) ->
            ( News apiData (showInfo index page), Cmd.none )

        ( StopHoveringEntity index, News apiData page ) ->
            ( News apiData (hideInfo index page), Cmd.none )

        ( ToggleReduce bool, Done page ) ->
            ( Done { page | reduceTags = bool }, Cmd.none )

        ( ToggleReduce bool, News apiData page ) ->
            ( News apiData { page | reduceTags = bool }, Cmd.none )

        ( NewNews result, News ind page ) ->
            result
                |> handleResult
                    (\article ->
                        ( News ind { page | text = article, prediction = Nothing, showingInfo = Set.empty }
                        , getPrediction page.selection article
                        )
                    )

        ( MoreNews, News apiData page ) ->
            ( News (nextResult apiData) page
            , getNews (nextResult apiData)
            )

        ( _, _ ) ->
            ( model, Cmd.none )


handleResult okHandler result =
    case result of
        Ok ok ->
            okHandler ok

        Err e ->
            ( Error e, Cmd.none )


updatePrediction model page =
    let
        okHandler pred =
            ( model { page | prediction = Just pred }
            , Cmd.none
            )
    in
    handleResult okHandler


updateLinks model page =
    handleResult (\links -> ( model { page | links = links }, Cmd.none ))



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions _ =
    Sub.none



-- VIEW


view model =
    case model of
        Done page ->
            Browser.Document "page"
                [ Element.layout []
                    (body page)
                ]

        Loading ->
            Browser.Document "loading"
                [ Element.layout []
                    (el [] (Element.text "loading"))
                ]

        Error e ->
            Browser.Document "error"
                [ Element.layout []
                    (el [] (Element.text (errorString e)))
                ]

        NoModels ->
            Browser.Document "error"
                [ Element.layout []
                    (el [] (Element.text "No models found"))
                ]

        News ind page ->
            Browser.Document "news"
                [ Element.layout []
                    (viewNews page)
                ]

        LoadingNews ->
            Browser.Document "loading"
                [ Element.layout []
                    (el [] (Element.text "loading"))
                ]


body : Page -> Element Msg
body { models, selection, text, prediction, reduceTags, showingInfo } =
    column [ width fill, spacing 30 ]
        [ row [ width fill ] [ reduceToggle reduceTags ]
        , resultView text selection prediction reduceTags showingInfo
        ]


viewNews : Page -> Element Msg
viewNews { selection, text, prediction, reduceTags, showingInfo } =
    column [ width fill, spacing 50, padding 150 ]
        [ row [ width fill ] [ reduceToggle reduceTags ]
        , el [ width fill, height fill ] (resultView text selection prediction reduceTags showingInfo)
        , el [ width fill, height fill ] (button { onPress = Just MoreNews, label = Element.text "Get more news!" })
        ]


button =
    Input.button
        [ centerX
        , Border.width 1
        , Background.color (Element.rgb 1 1 1)
        , Border.color (Element.rgb255 219 219 219)
        , Border.rounded 290486
        , padding 8
        , Element.mouseOver [ Border.color (Element.rgb 0 0 0) ]
        ]


reduceToggle reduceTags =
    Input.checkbox []
        { onChange = ToggleReduce
        , icon = Input.defaultCheckbox
        , checked = reduceTags
        , label = Input.labelLeft [] (Element.text "Reduce Tags")
        }


selectModel : String -> List String -> Element Msg
selectModel selection models =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Element.html (select [ onInput NewSelection ] (selection :: models |> List.map strToOption)))


strToOption str =
    option [] [ Html.text str ]


getSel sel docs =
    let
        get dict key =
            Dict.get key dict
    in
    sel |> Maybe.andThen (get docs)


resultView : String -> String -> Maybe Document -> Bool -> Set.Set Int -> Element Msg
resultView text selection prediction reduceTags showingInfo =
    let
        pos x y tag attrs =
            tag ([ SAttr.x (String.fromFloat x), SAttr.y (String.fromFloat y) ] ++ attrs)

        textStyle sz =
            SAttr.style ("font-size: " ++ String.fromInt sz ++ "px;font-family: 'Source Code Pro', monospace;")

        charWidth =
            12

        charHeight =
            15

        colorFromClass class =
            case class of
                "NAM-PER" ->
                    "#78CAD2"

                "NAM-FAC" ->
                    "#63595C"

                "NAM-LOC" ->
                    "#646881"

                "NAM-ORG" ->
                    "#62BEC1"

                "NAM-TTL" ->
                    "#5AD2F4"

                "NAM-GPE" ->
                    "#72DDF7"

                "NOM-PER" ->
                    "#F865B0"

                "NOM-FAC" ->
                    "#E637BF"

                "NOM-LOC" ->
                    "#FF928B"

                "NOM-ORG" ->
                    "#FEC3A6"

                "NOM-TTL" ->
                    "#FF3C38"

                "NOM-GPE" ->
                    "#BB8588"

                _ ->
                    "red"

        mark string =
            let
                length =
                    String.length string

                w =
                    length * charWidth + 8 |> String.fromInt

                height =
                    charHeight + 8

                h =
                    height |> String.fromFloat

                padding =
                    4
            in
            Svg.svg [ SAttr.width w, SAttr.height h, SAttr.viewBox ("0 0 " ++ w ++ " " ++ h) ]
                [ Svg.g []
                    [ pos 0
                        0
                        Svg.rect
                        [ SAttr.width w
                        , SAttr.height h
                        , SAttr.rx "5"
                        , SAttr.ry "5"
                        , SAttr.style ("fill:" ++ colorFromClass string ++ ";stroke:black;stroke-width:1;opacity:0.5")
                        ]
                        []
                    , pos padding (height - padding) Svg.text_ [ textStyle 20 ] [ Svg.text string ]
                    ]
                ]

        annotate : Int -> Int -> String -> List Entity -> List (Element Msg)
        annotate index origin string ent =
            let
                line begin end =
                    el [] (plain begin end)

                plain begin end =
                    Element.text (String.slice begin end string)

                entityImage image name =
                    Element.image [ width fill, Element.centerX, Border.width 1 ] { src = image, description = "Image of depicting " ++ name }

                entityLink maybeData label =
                    case maybeData of
                        Just { image, name, link } ->
                            Element.link
                                []
                                { url = link
                                , label =
                                    el
                                        [ Element.mouseOver [ Font.color (Element.rgb255 4 46 115) ]
                                        , Font.color (Element.rgb255 6 69 173)
                                        ]
                                        label
                                }

                        Nothing ->
                            label

                infoBox { name, image, description } =
                    Element.column
                        [ Element.centerX
                        , Background.color (Element.rgb 1 1 1)
                        , Border.rounded 5
                        , Border.width 1
                        , width (px 300)
                        , padding 5
                        , spacing 5
                        ]
                        [ el [] (Element.text name)
                        , el [] (entityImage image name)
                        , Element.paragraph [] [ Element.text description ]
                        ]

                marked { tag, wikidata } =
                    let
                        attributes maybeData =
                            case maybeData of
                                Just data ->
                                    Element.centerX
                                        :: (if Set.member index showingInfo then
                                                [ infoBox data |> Element.below ]

                                            else
                                                []
                                           )

                                Nothing ->
                                    [ Element.centerX ]
                    in
                    Element.column
                        [ Events.onMouseEnter (StartHoveringEntity index)
                        , Events.onMouseLeave (StopHoveringEntity index)
                        ]
                        [ el (attributes wikidata)
                            (entityLink wikidata
                                (plain tag.start tag.stop)
                            )
                        , el
                            [ Element.centerX ]
                            (mark tag.class |> Element.html)
                        ]
            in
            case ent of
                entity :: tail ->
                    line origin entity.tag.start :: marked entity :: annotate (index + 1) entity.tag.stop string tail

                [] ->
                    [ line origin (String.length string) ]

        viewAnnotations =
            Element.paragraph [ Font.family [ Font.typeface "Source Sans Pro", Font.sansSerif ] ]

        viewPrediction pred =
            case pred of
                Just document ->
                    document.reduced
                        |> annotate 0 0 document.text
                        |> viewAnnotations

                Nothing ->
                    Element.paragraph [ width fill, alignTop, Element.inFront (el [ width fill, alignTop ] spinner) ] [ Element.text text ]
    in
    el [ width fill, spacing 50 ]
        (viewPrediction prediction)


spinner =
    column [ centerX, alignTop ]
        [ Html.div [ class "lds-dual-ring" ] [] |> Element.html |> el [ centerX, padding 60 ]
        , el [ Font.center, width fill ] (Element.text "Predicting labels  ")
        ]


errorString error =
    case error of
        Http.BadBody str ->
            "BadBody: " ++ str

        Http.BadStatus code ->
            "BadStatus: " ++ String.fromInt code

        Http.BadUrl str ->
            "BadUrl: " ++ str

        Http.NetworkError ->
            "NetworkError"

        Http.Timeout ->
            "Timeout"



-- HTTP


localApi : String
localApi =
    UrlBuilder.absolute [ "models" ] []


getModels : Cmd Msg
getModels =
    Http.get
        { url = localApi
        , expect = Http.expectJson NewModels (list string)
        }


getPrediction : String -> String -> Cmd Msg
getPrediction model text =
    Http.post
        { url = UrlBuilder.absolute [ "predict" ] [ UrlBuilder.string "model" model ]
        , body = Http.jsonBody (Encode.string text)
        , expect = Http.expectJson NewPrediction documentDecoder
        }


getLinks : List String -> Cmd Msg
getLinks entities =
    Http.post
        { url = UrlBuilder.absolute [ "links" ] [ UrlBuilder.string "lang" "en" ]
        , body = entities |> Encode.list Encode.string |> Http.jsonBody
        , expect = Http.expectJson NewLinks (Decode.list Decode.string)
        }


documentDecoder : Decoder Document
documentDecoder =
    Decode.succeed Document
        |> required "text" string
        |> required "entities" (list tagDecoder)
        |> custom entityDecoder


tagDecoder =
    Decode.succeed Tag
        |> required "start" int
        |> required "stop" int
        |> required "class" string


entityDecoder =
    Decode.succeed (List.map2 initEntity)
        |> required "reduced" (list tagDecoder)
        |> required "targets" (list (Decode.maybe wikidataTupleDecoder))


newsApi : String -> String -> String -> String
newsApi lang query apiKey =
    UrlBuilder.crossOrigin "https://newsapi.org/"
        [ "v2", "everything" ]
        [ UrlBuilder.string "language" lang
        , UrlBuilder.string "q" query
        , UrlBuilder.string "apiKey" apiKey
        ]


getNews : ApiData -> Cmd Msg
getNews { query, index, key } =
    Http.get
        { url = newsApi "en" query key
        , expect = Http.expectJson NewNews (newsDecoder index)
        }


newsDecoder ind =
    field "articles" (Decode.index ind (field "description" string))


wikidataTupleDecoder =
    Decode.succeed Wikidata
        |> required "entity" string
        |> required "image" string
        |> required "name" string
        |> required "sitelink" string
        |> required "entityDescription" string
